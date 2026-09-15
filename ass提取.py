import re
import json


def clean_ass_text(text):
    """清洗 ASS 字幕里的样式标签和特殊符号"""
    # 1. 移除 {\...} 格式的 ASS 特效/样式标签
    text = re.sub(r'\{.*?}', '', text)
    # 2. 将 ASS 的换行符 \N 或 \n 替换为空格或句号
    text = re.sub(r'\\[Nn]', ' ', text)
    # 3. 去除首尾多余空格
    return text.strip()


def parse_ass_to_json(ass_filepath, output_json_filepath):
    dialogues = []

    with open(ass_filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
        lines = f.readlines()

    in_events = False
    format_headers = []

    for line in lines:
        line = line.strip()

        # 标记是否进入到了事件/台词区块
        if line.startswith('[Events]'):
            in_events = True
            continue

        if in_events:
            # 读取 Format 列头定义，获取各字段的索引位置
            if line.startswith('Format:'):
                headers_raw = line.split('Format:')[1].strip()
                format_headers = [h.strip() for h in headers_raw.split(',')]

            # 读取具体台词行
            elif line.startswith('Dialogue:'):
                content = line.split('Dialogue:', 1)[1].strip()
                # 按照 Format 要求的列数拆分（最后一列 Text 可能包含逗号，所以要限制 split 深度）
                fields = content.split(',', len(format_headers) - 1)

                if len(fields) == len(format_headers):
                    field_dict = dict(zip(format_headers, fields))

                    cleaned_text = clean_ass_text(field_dict.get('Text', ''))

                    # 如果清洗后台词不为空，则存入列表
                    if cleaned_text:
                        item = {
                            "start": field_dict.get('Start', ''),
                            "end": field_dict.get('End', ''),
                            "speaker": field_dict.get('Name', '').strip(),  # 说话人/角色名
                            "style": field_dict.get('Style', '').strip(),  # 样式名称（有时用来区分角色）
                            "text": cleaned_text
                        }
                        dialogues.append(item)

    # 保存为标准 JSON 文件
    with open(output_json_filepath, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)

    print(f"成功导出 {len(dialogues)} 条台词到 {output_json_filepath}")


# ================= 使用示例 =================
# 替换为你的 .ass 文件路径
ass_file = "/Users/hank/PycharmProjects/PythonProject3/Cosmic.Princess.Kaguya.2026.1080p.NF.WEB-DL.MULTi.DDP5.1.H.264-@shenghuo2-v3.ass"
json_file = "kaguya_dialogues.json"

parse_ass_to_json(ass_file, json_file)