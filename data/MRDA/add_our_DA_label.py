# 定义核心映射字典：原始DA标签 → 12类标签缩写
da_to_eml_abbr = {
    # Assertion_Persuasive (ap)
    's': 'ap',
    'df': 'ap',
    # Assertion_Committal (ac)
    'cc': 'ac',
    # Directive_Command (dc)
    'co': 'dc',
    # Directive_Offer (do)
    'cs': 'do',
    # Interrogative_Leading (il)
    'qy': 'il',
    'g': 'il',
    # Interrogative_Challenging (ic)
    'qh': 'ic',
    # Interrogative_Neutral (in)
    'qw': 'in',
    'qrr': 'in',
    'qr': 'in',
    'qo': 'in',
    'bh': 'in',
    # Position_Accept (pa)
    'bk': 'pa',
    'aa': 'pa',
    'na': 'pa',
    'aap': 'pa',
    # Position_Reject (pr)
    'ar': 'pr',
    'nd': 'pr',
    'ng': 'pr',
    'bsc': 'pr',
    # Signal_Positive (sp)
    'j': 'sp',
    'ft': 'sp',
    'fw': 'sp',
    # Signal_Negative (sn)
    'bd': 'sn',
    'fa': 'sn',
    'by': 'sn',
    # Others (o) - 所有未匹配的标签默认归为Others
    'b': 'o',
    'fh': 'o',
    'e': 'o',
    '%': 'o',
    'rt': 'o',
    'fg': 'o',
    'ba': 'o',
    'bu': 'o',
    'd': 'o',
    '2': 'o',
    'no': 'o',
    'h': 'o',
    'fe': 'o',
    'm': 'o',
    't': 'o',
    'br': 'o',
    'tc': 'o',
    'r': 'o',
    't1': 'o',
    't3': 'o',
    'arp': 'o',
    'bs': 'o',
    'f': 'o',
    'bc': 'o',
    'am': 'o'  # 按建议将Maybe归为Others
}


def add_eml_da_label(input_file="backup/val_set.txt", output_file="val_set.txt"):
    """
    为full_set.txt添加12类EML-DA标签缩写
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding = 'utf-8') as f_in, \
            open(output_file, 'w', encoding = 'utf-8') as f_out:

        line_count = 0
        processed_count = 0

        for line in f_in:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            line_count += 1

            # 按|分割行内容（处理格式：fe016|okay.|F|fg|fg）
            parts = line.split('|')
            if len(parts) < 5:  # 格式校验，避免异常行
                print(f"警告：第{line_count}行格式异常，跳过 → {line}")
                continue

            # 提取第三个|后的标签（即parts[3]，索引从0开始）
            original_da_label = parts[3].strip()

            # 映射到12类标签缩写（默认Others）
            eml_abbr = da_to_eml_abbr.get(original_da_label, 'o')

            # 拼接新行（原内容 + | + 标签缩写）
            new_line = '|'.join(parts) + f'|{eml_abbr}'
            f_out.write(new_line + '\n')
            processed_count += 1

        print(f"\n处理完成！")
        print(f"总行数：{line_count} | 成功处理行数：{processed_count}")
        print(f"输出文件：{output_file}")


# 执行脚本
if __name__ == "__main__":
    add_eml_da_label()
