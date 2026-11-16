#!/usr/bin/env python3
"""Generate chat_template.jinja for DeepSeek-V4-Flash-0731 from its official encoder.

0731 ships no jinja, only encoding/encoding_dsv4.py. The offline-OPD rebuild
(teacher_trajectory.py) renders through tokenizer.apply_chat_template, so this
writes a jinja that reproduces encode_messages(..., thinking_mode="thinking")
byte-for-byte for the shapes the GLM scaffold produces:
  system(+tools) / user / assistant(reasoning, content, tool_calls) / tool.

Two deliberate choices, both matching the official encoder's tool-calling path:
  * reasoning is NEVER dropped. encode_messages disables drop_thinking whenever a
    message carries tools (README: "tool-calling conversations require full
    context"); our real prompts always carry tools. The rebuild's probe renders
    have no tools, and dropping there would break the prefix property
    _template_turn_deltas relies on.
  * consecutive tool messages merge into ONE <|User|> turn whose blocks are
    "\n\n"-joined <tool_result>...</tool_result> (merge_tool_messages).
The literal tool-instruction text is taken from TOOLS_TEMPLATE itself, so it
cannot drift from the encoder.

Usage: python3 make_dsv4_0731_chat_template.py <0731 model dir> <out .jinja>
"""
import json
import sys
from pathlib import Path


def main() -> None:
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(model_dir / "encoding"))
    import encoding_dsv4 as e  # noqa: E402

    tools_text = e.TOOLS_TEMPLATE.format(
        tool_schemas="\x00SCHEMAS\x00",
        dsml_token=e.dsml_token,
        thinking_start_token=e.thinking_start_token,
        thinking_end_token=e.thinking_end_token,
    )
    tools_head, tools_tail = tools_text.split("\x00SCHEMAS\x00")
    efforts = {k: v for k, v in e.REASONING_EFFORT_PROMPTS.items()}
    D = e.dsml_token

    def lit(s: str) -> str:
        # JSON string literal == valid jinja string literal for our content.
        return json.dumps(s, ensure_ascii=False)

    j = []
    a = j.append
    a("{%- set efforts = " + json.dumps(efforts, ensure_ascii=False) + " -%}")
    a("{%- set effort = reasoning_effort if reasoning_effort is defined and reasoning_effort else 'low' -%}")
    a("{{- " + lit(e.bos_token) + " -}}")
    a("{{- efforts[effort] -}}")
    a("{%- set ns = namespace(in_user=false) -%}")
    a("{%- for m in messages -%}")
    a("{%- set nxt = messages[loop.index0 + 1] if not loop.last else none -%}")
    # ---------------------------------------------------------------- system
    a("{%- if m.role == 'system' -%}")
    a("{{- m.content or '' -}}")
    a("{%- if tools -%}")
    a("{{- '\\n\\n' + " + lit(tools_head) + " -}}")
    a("{%- for t in tools -%}{%- if not loop.first -%}{{- '\\n' -}}{%- endif -%}"
      "{{- (t.function if t.function is defined else t) | tojson(ensure_ascii=False) -}}{%- endfor -%}")
    a("{{- " + lit(tools_tail) + " -}}")
    a("{%- endif -%}")
    # ---------------------------------------------------------------- user / tool
    a("{%- elif m.role == 'user' or m.role == 'tool' -%}")
    a("{%- if not ns.in_user -%}{{- " + lit(e.USER_SP_TOKEN) + " -}}{%- set ns.in_user = true -%}"
      "{%- else -%}{{- '\\n\\n' -}}{%- endif -%}")
    a("{%- if m.role == 'tool' -%}{{- '<tool_result>' + (m.content or '') + '</tool_result>' -}}"
      "{%- else -%}{{- m.content or '' -}}{%- endif -%}")
    a("{%- if nxt is none or (nxt.role != 'user' and nxt.role != 'tool') -%}")
    a("{%- set ns.in_user = false -%}")
    a("{%- if (nxt is not none and nxt.role == 'assistant') or (nxt is none and add_generation_prompt) -%}")
    a("{{- " + lit(e.ASSISTANT_SP_TOKEN + e.thinking_start_token) + " -}}")
    a("{%- endif -%}")
    a("{%- endif -%}")
    # ---------------------------------------------------------------- assistant
    a("{%- elif m.role == 'assistant' -%}")
    a("{{- (m.reasoning_content or '') + " + lit(e.thinking_end_token) + " + (m.content or '') -}}")
    a("{%- if m.tool_calls -%}")
    a("{{- '\\n\\n<" + D + "tool_calls>\\n' -}}")
    a("{%- for tc in m.tool_calls -%}")
    a("{%- set f = tc.function if tc.function is defined else tc -%}")
    a("{%- if not loop.first -%}{{- '\\n' -}}{%- endif -%}")
    a("{{- '<" + D + "invoke name=\"' + f.name + '\">\\n' -}}")
    a("{%- set args = f.arguments -%}")
    a("{%- if args is string -%}{{- '<" + D + "parameter name=\"arguments\" string=\"true\">' + args + '</" + D + "parameter>' -}}")
    a("{%- else -%}{%- for k, v in args.items() -%}{%- if not loop.first -%}{{- '\\n' -}}{%- endif -%}"
      "{%- if v is string -%}{{- '<" + D + "parameter name=\"' + k + '\" string=\"true\">' + v + '</" + D + "parameter>' -}}"
      "{%- else -%}{{- '<" + D + "parameter name=\"' + k + '\" string=\"false\">' + (v | tojson(ensure_ascii=False)) + '</" + D + "parameter>' -}}"
      "{%- endif -%}{%- endfor -%}{%- endif -%}")
    a("{{- '\\n</" + D + "invoke>' -}}")
    a("{%- endfor -%}")
    a("{{- '\\n</" + D + "tool_calls>' -}}")
    a("{%- endif -%}")
    a("{{- " + lit(e.eos_token) + " -}}")
    a("{%- endif -%}")
    a("{%- endfor -%}")
    out.write_text("".join(j) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
