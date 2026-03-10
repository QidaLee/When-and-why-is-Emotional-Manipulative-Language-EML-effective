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

# 标签全称映射（用于展示更清晰）
abbr_to_full = {
    'ap': 'Assertion_Persuasive',
    'ac': 'Assertion_Committal',
    'dc': 'Directive_Command',
    'do': 'Directive_Offer',
    'il': 'Interrogative_Leading',
    'ic': 'Interrogative_Challenging',
    'in': 'Interrogative_Neutral',
    'pa': 'Position_Accept',
    'pr': 'Position_Reject',
    'sp': 'Signal_Positive',
    'sn': 'Signal_Negative',
    'o': 'Others'
}


def add_eml_da_label(input_file="backup/val_set.txt", output_file="val_set.txt"):
    """
    为full_set.txt添加12类EML-DA标签缩写，并统计标签分布
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    # 初始化标签计数器
    label_counter = {
        'ap': 0, 'ac': 0, 'dc': 0, 'do': 0,
        'il': 0, 'ic': 0, 'in': 0, 'pa': 0,
        'pr': 0, 'sp': 0, 'sn': 0, 'o': 0
    }

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

            # 提取第四个|后的标签（即parts[4]，索引从0开始）
            original_da_label = parts[4].strip()

            # 映射到12类标签缩写（默认Others）
            eml_abbr = da_to_eml_abbr.get(original_da_label, 'o')

            # 更新标签计数器
            label_counter[eml_abbr] += 1

            # 拼接新行（原内容 + | + 标签缩写）
            new_line = '|'.join(parts) + f'|{eml_abbr}'
            f_out.write(new_line + '\n')
            processed_count += 1

        # 打印基础处理信息
        print(f"\n===== 处理完成 ======")
        print(f"总行数：{line_count} | 成功处理行数：{processed_count}")
        print(f"输出文件：{output_file}")

        # 打印标签分布统计
        print(f"\n===== 标签分布统计 =====")
        print(f"{'缩写':<4} {'全称':<25} {'数量':<8} {'占比':<8}")
        print("-" * 50)
        total_valid = sum(label_counter.values())
        # 按数量降序排序展示
        sorted_labels = sorted(label_counter.items(), key = lambda x: x[1], reverse = True)
        for abbr, count in sorted_labels:
            full_name = abbr_to_full.get(abbr, abbr)
            ratio = (count / total_valid) * 100 if total_valid > 0 else 0
            print(f"{abbr:<4} {full_name:<25} {count:<8} {ratio:.2f}%")

        # 验证总数（避免统计错误）
        print("-" * 50)
        print(f"统计总数：{total_valid} | 匹配处理行数：{processed_count}")
        if total_valid != processed_count:
            print("⚠️  警告：统计总数与处理行数不一致，请检查数据！")


# 执行脚本
if __name__ == "__main__":
    add_eml_da_label()