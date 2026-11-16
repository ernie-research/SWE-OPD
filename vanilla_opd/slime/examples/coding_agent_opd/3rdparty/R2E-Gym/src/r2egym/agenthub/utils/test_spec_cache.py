import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from r2egym.agenthub.utils.log import get_logger

logger = get_logger(__name__)


class TestSpecCache:
    """
    缓存管理器，用于存储和检索TestSpec对象，避免重复的GitHub API调用
    """
    
    def __init__(self, cache_dir: str = "./swebench_test_spec_cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # logger.info(f"TestSpecCache initialized with directory: {self.cache_dir}")
    
    def _get_cache_key(self, ds: Dict[str, Any]) -> str:
        """
        根据数据集条目生成缓存键
        
        Args:
            ds: 数据集条目
            
        Returns:
            唯一的缓存键字符串
        """
        # 优先使用instance_id
        if 'instance_id' in ds:
            return ds['instance_id']
        
        # 备选方案：基于repo和commit生成唯一键
        repo = ds.get('repo', ds.get('repo_name', 'unknown'))
        base_commit = ds.get('base_commit', ds.get('commit', 'unknown'))
        # 使用hash避免文件名过长或包含非法字符
        unique_str = f"{repo}_{base_commit}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存文件的完整路径
        """
        # 使用安全的文件名
        safe_key = "".join(c for c in cache_key if c.isalnum() or c in '-_.')
        return self.cache_dir / f"{safe_key}.pkl"
    
    def exists(self, ds: Dict[str, Any]) -> bool:
        """
        检查指定数据集条目的缓存是否存在
        
        Args:
            ds: 数据集条目
            
        Returns:
            如果缓存存在返回True，否则返回False
        """
        cache_key = self._get_cache_key(ds)
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()
    
    def get(self, ds: Dict[str, Any]) -> Optional[Any]:
        """
        从缓存中获取TestSpec对象
        
        Args:
            ds: 数据集条目
            
        Returns:
            如果找到缓存返回TestSpec对象，否则返回None
        """
        cache_key = self._get_cache_key(ds)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                test_spec = pickle.load(f)
            logger.debug(f"Cache hit for {cache_key}")
            return test_spec
        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
            # 删除损坏的缓存文件
            try:
                cache_path.unlink()
            except:
                pass
            return None
    
    def set(self, ds: Dict[str, Any], test_spec: Any) -> bool:
        """
        将TestSpec对象保存到缓存
        
        Args:
            ds: 数据集条目
            test_spec: 要缓存的TestSpec对象
            
        Returns:
            保存成功返回True，否则返回False
        """
        cache_key = self._get_cache_key(ds)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(test_spec, f)
            logger.debug(f"Cache saved for {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache for {cache_key}: {e}")
            return False
    
    def clear(self) -> int:
        """
        清除所有缓存文件
        
        Returns:
            删除的文件数量
        """
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cache files")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        return count
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": int(round(total_size / (1024 * 1024)))
        }


def preload_test_specs(
    ds_entries: List[Dict[str, Any]], 
    max_workers: int = 5,
    cache_dir: str = "./test_spec_cache"
) -> Dict[str, int]:
    """
    预先拉取并缓存所有SWE-bench验证数据集的TestSpec
    
    Args:
        ds_entries: 数据集条目列表
        max_workers: 并行拉取的最大工作线程数
        cache_dir: 缓存目录
        
    Returns:
        包含统计信息的字典：{'total': int, 'cached': int, 'fetched': int, 'failed': int}
    """
    cache = TestSpecCache(cache_dir)
    
    # 过滤出SWE-bench相关的条目
    swebench_entries = []
    for ds_entry in ds_entries:
        docker_image = ds_entry.get('docker_image', '')
        if 'swebench' in docker_image:
            swebench_entries.append(ds_entry)
    
    if not swebench_entries:
        logger.info("No SWE-bench entries found, skipping test spec preload")
        return {'total': 0, 'cached': 0, 'fetched': 0, 'failed': 0}
    
    # 统计已缓存和需要拉取的条目
    to_fetch = []
    cached_count = 0
    
    for ds_entry in swebench_entries:
        if cache.exists(ds_entry):
            cached_count += 1
        else:
            to_fetch.append(ds_entry)
    
    logger.info(f"TestSpec preload: {len(swebench_entries)} total, {cached_count} cached, {len(to_fetch)} to fetch")
    
    if not to_fetch:
        logger.info("All test specifications already cached")
        return {
            'total': len(swebench_entries),
            'cached': cached_count,
            'fetched': 0,
            'failed': 0
        }
    
    logger.info(f"Pre-fetching {len(to_fetch)} test specifications with {max_workers} workers...")
    
    def fetch_one(ds_entry: Dict[str, Any]) -> bool:
        """拉取单个test spec的内部函数"""
        cache_key = cache._get_cache_key(ds_entry)
        try:
            logger.debug(f"Fetching test spec for {cache_key}")
            # 动态导入make_test_spec避免循环导入
            try:
                from r2egym.agenthub.trajectory.swebench_utils import make_test_spec
            except ImportError:
                from swebench.harness.test_spec.test_spec import make_test_spec
            
            test_spec = make_test_spec(ds_entry)
            success = cache.set(ds_entry, test_spec)
            if success:
                logger.debug(f"Successfully cached test spec for {cache_key}")
            else:
                logger.error(f"Failed to cache test spec for {cache_key}")
            return success
        except Exception as e:
            logger.error(f"Failed to fetch test spec for {cache_key}: {e}")
            return False
    
    # 并行拉取
    fetched_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_entry = {
            executor.submit(fetch_one, ds_entry): ds_entry 
            for ds_entry in to_fetch
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_entry):
            try:
                success = future.result()
                if success:
                    fetched_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                ds_entry = future_to_entry[future]
                cache_key = cache._get_cache_key(ds_entry)
                logger.error(f"Exception during fetch for {cache_key}: {e}")
                failed_count += 1
    
    stats = {
        'total': len(swebench_entries),
        'cached': cached_count,
        'fetched': fetched_count,
        'failed': failed_count
    }
    
    logger.info(f"TestSpec preload completed: {stats}")
    return stats 