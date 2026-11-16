"""module for response helper which handle"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
import traceback

import numpy as np
from r2egym.agenthub.utils.log import get_logger
logger = get_logger(__name__)


class OpenAIResponseHandler:
    """OpenAI Response Handler"""

    def __init__(self, output_dir: str = None):
        """
        初始化处理器

        Args:
            output_file: 输出的JSONL文件路径
        """
        if output_dir is None:
            output_dir = os.environ.get("CODE_BENCH_TASK_DIR", "./output/")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "openai_responses.jsonl")
        self.summary_file = os.path.join(output_dir, "openai_responses_summary.txt")
        self.responses = []

    def add_response(self, response: Dict[str, Any]) -> None:
        """
        添加一个新的响应数据并保存到文件

        Args:
            response: OpenAI API的响应数据字典
        """
        self.responses.append(response)

        # 将数据追加到JSONL文件
        with open(self.output_file, "a+", encoding="utf-8") as f:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")

    def load_responses_from_file(self, file_path: Optional[str] = None) -> None:
        """
        从文件加载响应数据

        Args:
            file_path: 输入的JSONL文件路径，默认使用输出文件路径
        """
        file_path = file_path or self.output_file

        if not os.path.exists(file_path):
            logger.info(f"文件 {file_path} 不存在")
            return

        self.responses = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.responses.append(json.loads(line))

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计信息

        Returns:
            包含各种统计信息的字典
        """
        if not self.responses:
            return {"error": "没有可用的响应数据"}

        valid_responses = []
        completion_tokens = []
        reasoning_tokens = []
        finish_reasons = []

        dead_loop_count = 0

        for response in self.responses:

            # 确保响应格式正确
            if not response.get("choices", []):
                continue

            usage = response["usage"]
            valid_responses.append(response)

            if "text" in response["choices"][0]:
                response_content = response["choices"][0]["text"]
            else:
                response_content = response["choices"][0]["message"]["content"]
            
            dead_loop_count += check_dead_loop(response_content)

            if not usage:
                continue

            # 收集completion_tokens
            if usage.get("completion_tokens", None):
                completion_tokens.append(usage["completion_tokens"])

            # 收集reasoning_tokens
            if usage.get("completion_tokens_details", None) and usage[
                "completion_tokens_details"
            ].get("reasoning_tokens", None):
                reasoning_tokens.append(
                    usage["completion_tokens_details"]["reasoning_tokens"]
                )

            # 收集finish_reasons
            if "choices" in response and response["choices"]:
                for choice in response["choices"]:
                    if "finish_reason" in choice:
                        finish_reasons.append(choice["finish_reason"])

        # 计算平均值和分布
        stats = {
            "total_responses": len(valid_responses),
            "average_completion_tokens": (
                sum(completion_tokens) / len(completion_tokens)
                if completion_tokens
                else 0
            ),
            "average_reasoning_tokens": (
                sum(reasoning_tokens) / len(reasoning_tokens) if reasoning_tokens else 0
            ),
            "finish_reason_distribution": self._calculate_distribution(finish_reasons),
            "dead_loop_count": dead_loop_count / len(valid_responses),
        }

        return stats

    def _calculate_distribution(self, items: List[Any]) -> Dict[str, Any]:
        """
        计算列表项的分布情况

        Args:
            items: 项目列表

        Returns:
            包含分布信息的字典
        """
        if not items:
            return {}

        counter = Counter(items)
        total = len(items)

        distribution = {
            "counts": dict(counter),
            "percentages": {k: round(v / total * 100, 2) for k, v in counter.items()},
        }

        return distribution

    def finalize_summary(self) -> str:
        """
        获取统计摘要的字符串表示

        Returns:
            格式化的统计摘要字符串
        """
        try:
            self.load_responses_from_file()
            stats = self.calculate_statistics()

            if "error" in stats:
                return stats["error"]

            summary = [
                f"总响应数: {stats['total_responses']}",
                f"平均completion_tokens: {stats['average_completion_tokens']:.2f}",
                f"平均reasoning_tokens: {stats['average_reasoning_tokens']:.2f}",
                f"恶劣重复比例: {stats['dead_loop_count']:.2f}",
                f"\n完成原因分布:",
            ]

            if stats["finish_reason_distribution"]:
                for reason, percentage in stats["finish_reason_distribution"][
                    "percentages"
                ].items():
                    count = stats["finish_reason_distribution"]["counts"][reason]
                    summary.append(f"  - {reason}: {count}次 ({percentage}%)")
            else:
                summary.append("  无可用数据")

        except Exception as e:
            # print stack trace
            print("错误信息：", e)
            traceback.print_exc()
            summary = "无"

        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))

        return "\n".join(summary)
    
    def update_summary_into_strategy_score_file(self, strategy_score_file: str):
        """
        将策略评分更新到策略评分文件中。

        Args:
            strategy_score_file (str): 策略评分文件的路径。

        Returns:
            None

        Raises:
            FileNotFoundError: 如果策略评分文件不存在。
            json.JSONDecodeError: 如果策略评分文件不是有效的 JSON 文件。
            Exception: 更新策略评分文件时发生其他错误。

        """
        try:
            # 读取文件内容
            with open(strategy_score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 读取 Response 整体分布
            self.load_responses_from_file()
            metrics = self.calculate_statistics()
            for metric_field, value in metrics.items():
                if not isinstance(value, (float, int)):
                    continue
                data["score"].append({
                    "metric_type": metric_field,
                    "score": value
                })
            
            # 添加 finish reason
            for reason, value in metrics["finish_reason_distribution"]["percentages"].items():
                data["score"].append({
                    "metric_type": f"finish_reason-{reason}",
                    "score": value
                })
            
            # write data into strategy_score_file
            with open(strategy_score_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except FileNotFoundError:
            logger.info(f"文件 {strategy_score_file} 未找到")
        except json.JSONDecodeError:
            logger.info(f"文件 {strategy_score_file} 不是有效的 JSON 文件")
        except Exception as e:
            logger.info(f"更新文件 {strategy_score_file} 时出错: {e}")
    

def check_dead_loop(response: str) -> bool:
    """检查是否存在死循环"""
    for sent in response.split('\n'):
        if response.count(sent) >= 10 and len(sent) > 30:
            return 1
    
    return 0
    
