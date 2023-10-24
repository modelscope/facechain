from __future__ import annotations
import html
import re
import traceback
from typing import List, Tuple

import json
import markdown
from gradio.components import Chatbot as ChatBotBase
ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

class ChatBot(ChatBotBase):

    def normalize_markdown(self, bot_message):
        lines = bot_message.split("\n")
        normalized_lines = []
        inside_list = False

        for i, line in enumerate(lines):
            if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
                if not inside_list and i > 0 and lines[i - 1].strip() != "":
                    normalized_lines.append("")
                inside_list = True
                normalized_lines.append(line)
            elif inside_list and line.strip() == "":
                if i < len(lines) - 1 and not re.match(r"^(\d+\.|-|\*|\+)\s",
                                                       lines[i + 1].strip()):
                    normalized_lines.append(line)
                continue
            else:
                inside_list = False
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def convert_markdown(self, bot_message):
        if bot_message.count('```') % 2 != 0:
            bot_message += '\n```'

        bot_message = self.normalize_markdown(bot_message)

        result = markdown.markdown(
            bot_message,
            extensions=[
                'toc', 'extra', 'tables', 'markdown_katex', 'codehilite',
                'mdx_truly_sane_lists', 'markdown_cjk_spacing.cjk_spacing',
                'pymdownx.magiclink'
            ],
            extension_configs={
                'markdown_katex': {
                    'no_inline_svg': True,  # fix for WeasyPrint
                    'insert_fonts_css': True,
                },
                'codehilite': {
                    'linenums': False,
                    'guess_lang': True
                },
                'mdx_truly_sane_lists': {
                    'nested_indent': 2,
                    'truly_sane': True,
                }
            })
        result = "".join(result)
        return result

    def convert_bot_message(self, bot_message):

        # 兼容老格式
        chunks = bot_message.split('<extra_id_0>')
        if len(chunks) > 1:
            new_bot_message = ''
            for idx, chunk in enumerate(chunks):
                new_bot_message += chunk
                if idx % 2 == 0:
                    if idx != len(chunks) - 1:
                        new_bot_message += '<|startofthink|>'
                else:
                    new_bot_message += '<|endofthink|>'

            bot_message = new_bot_message

        start_pos = 0
        result = ''
        find_json_pattern = re.compile(r'{[\s\S]+}')
        START_OF_THINK_TAG, END_OF_THINK_TAG = '<|startofthink|>', '<|endofthink|>'
        START_OF_EXEC_TAG, END_OF_EXEC_TAG = '<|startofexec|>', '<|endofexec|>'
        while start_pos < len(bot_message):
            try:
                start_of_think_pos = bot_message.index(START_OF_THINK_TAG,
                                                       start_pos)
                end_of_think_pos = bot_message.index(END_OF_THINK_TAG,
                                                     start_pos)
                if start_pos < start_of_think_pos:
                    result += self.convert_markdown(
                        bot_message[start_pos:start_of_think_pos])
                think_content = bot_message[start_of_think_pos
                                            + len(START_OF_THINK_TAG
                                                  ):end_of_think_pos].strip()
                json_content = find_json_pattern.search(think_content)
                think_content = json_content.group(
                ) if json_content else think_content
                try:
                    think_node = json.loads(think_content)
                    plugin_name = think_node.get(
                        'plugin_name',
                        think_node.get('plugin',
                                       think_node.get('api_name', 'unknown')))
                    summary = f'选择插件【{plugin_name}】，调用处理中...'
                    del think_node['url']
                    #think_node.pop('url', None)

                    detail = f'```json\n\n{json.dumps(think_node,indent=3,ensure_ascii=False)}\n\n```'
                except Exception:
                    summary = f'思考中...'
                    detail = think_content
                    # traceback.print_exc()
                    # detail += traceback.format_exc()
                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'
                #print(f'detail:{detail}')
                start_pos = end_of_think_pos + len(END_OF_THINK_TAG)
            except Exception:
                # result += traceback.format_exc()
                break
                # continue

            try:
                start_of_exec_pos = bot_message.index(START_OF_EXEC_TAG,
                                                      start_pos)
                end_of_exec_pos = bot_message.index(END_OF_EXEC_TAG, start_pos)
                # print(start_of_exec_pos)
                # print(end_of_exec_pos)
                # print(bot_message[start_of_exec_pos:end_of_exec_pos])
                # print('------------------------')
                if start_pos < start_of_exec_pos:
                    result += self.convert_markdown(
                        bot_message[start_pos:start_of_think_pos])
                exec_content = bot_message[start_of_exec_pos
                                           + len(START_OF_EXEC_TAG
                                                 ):end_of_exec_pos].strip()
                try:
                    summary = f'完成插件调用.'
                    detail = f'```json\n\n{exec_content}\n\n```'
                except Exception:
                    pass

                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'

                start_pos = end_of_exec_pos + len(END_OF_EXEC_TAG)
            except Exception:
                # result += traceback.format_exc()
                continue
        if start_pos < len(bot_message):
            result += self.convert_markdown(bot_message[start_pos:])
        result += ALREADY_CONVERTED_MARK
        return result

    
    def postprocess(
        self,
        message_pairs: list[list[str | tuple[str] | tuple[str, str] | None] | tuple],
    ) -> list[list[str | dict | None]]:
        """
        Parameters:
            message_pairs: List of lists representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.  It can also be a tuple whose first element is a string or pathlib.Path filepath or URL to an image/video/audio, and second (optional) element is the alt text, in which case the media file is displayed. It can also be None, in which case that message is not displayed.
        Returns:
            List of lists representing the message and response. Each message and response will be a string of HTML, or a dictionary with media information. Or None if the message is not to be displayed.
        """
        if message_pairs is None:
            return []
        processed_messages = []
        for message_pair in message_pairs:
            assert isinstance(
                message_pair, (tuple, list)
            ), f"Expected a list of lists or list of tuples. Received: {message_pair}"
            assert (
                len(message_pair) == 2
            ), f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}"
            if isinstance(message_pair[0], tuple) or isinstance(message_pair[1], tuple):
                processed_messages.append(
                [
                    self._postprocess_chat_messages(message_pair[0]),
                    self._postprocess_chat_messages(message_pair[1]),
                ]
            )
            else:
    # 处理不是元组的情况
                user_message, bot_message = message_pair

                if user_message and not user_message.endswith(
                        ALREADY_CONVERTED_MARK):
                    convert_md = self.convert_markdown(html.escape(user_message))
                    user_message = f"<p style=\"white-space:pre-wrap;\">{convert_md}</p>" + ALREADY_CONVERTED_MARK
                if bot_message and not bot_message.endswith(
                        ALREADY_CONVERTED_MARK):
                    bot_message = self.convert_bot_message(bot_message)
                processed_messages.append(
                    [
                        user_message,
                        bot_message,
                    ]
                )
            
        return processed_messages
