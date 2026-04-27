# -------------------------- DA Label Mapping (12-class EML-DA) --------------------------
# Core mapping: Original MRDA DA label → 12-class EML abbreviation
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
    # Others (o) - All unmatched labels default to Others
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
    'am': 'o'
}

# Full label names for display
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


# -------------------------- Main Function: Merge Utterances by Speaker --------------------------
def merge_speaker_utterance(input_file="backup/val_set.txt", output_file="val_set_merged.txt"):
    """
    Merge consecutive utterances from the same speaker into one full turn.
    Output format: speaker | merged_utterance | da_labels (comma-separated, unique)
    """
    # Initialize label counter
    label_counter = {key: 0 for key in abbr_to_full.keys()}

    with open(input_file, 'r', encoding = 'utf-8') as f_in, \
            open(output_file, 'w', encoding = 'utf-8') as f_out:

        # Track current speaker state
        current_speaker = None
        merged_text = []
        merged_labels = set()

        total_lines = 0
        merged_turns = 0

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            # Split line by |
            parts = line.split('|')
            if len(parts) < 5:
                print(f"Warning: Invalid format at line {total_lines}, skipped.")
                continue

            # Extract fields
            speaker = parts[0].strip()
            utterance = parts[1].strip()
            orig_da = parts[4].strip()

            # Map to 12-class label
            eml_label = da_to_eml_abbr.get(orig_da, 'o')

            # ------------------------------
            # Merge logic for same speaker
            # ------------------------------
            if speaker == current_speaker:
                # Continue merging
                merged_text.append(utterance)
                merged_labels.add(eml_label)
            else:
                # Speaker changed: write previous merged turn
                if current_speaker is not None:
                    full_text = ' '.join(merged_text)
                    label_str = ','.join(sorted(merged_labels))
                    f_out.write(f"{current_speaker}|{full_text}|{label_str}\n")
                    merged_turns += 1

                    # Update label count
                    for lab in merged_labels:
                        label_counter[lab] += 1

                # Reset for new speaker
                current_speaker = speaker
                merged_text = [utterance]
                merged_labels = {eml_label}

        # Write the last speaker turn
        if current_speaker is not None:
            full_text = ' '.join(merged_text)
            label_str = ','.join(sorted(merged_labels))
            f_out.write(f"{current_speaker}|{full_text}|{label_str}\n")
            merged_turns += 1
            for lab in merged_labels:
                label_counter[lab] += 1

    # ------------------------------
    # Print summary statistics
    # ------------------------------
    print("\n===== Processing Completed =====")
    print(f"Original lines processed: {total_lines}")
    print(f"Merged speaker turns:      {merged_turns}")
    print(f"Output saved to:           {output_file}")

    print("\n===== Label Distribution =====")
    print(f"{'Abbr':<5} {'Full Name':<28} {'Count':<6}")
    print("-" * 45)
    for abbr, count in label_counter.items():
        print(f"{abbr:<5} {abbr_to_full[abbr]:<28} {count:<6}")


# -------------------------- Run the Script --------------------------
if __name__ == "__main__":
    # Process validation set
    merge_speaker_utterance(
        input_file = "full_set.txt",
        output_file = "full_set_merged.txt"
    )

    # Uncomment to process train / test sets
    # merge_speaker_utterance("backup/train_set.txt", "train_set_merged.txt")
    # merge_speaker_utterance("backup/test_set.txt", "test_set_merged.txt")
