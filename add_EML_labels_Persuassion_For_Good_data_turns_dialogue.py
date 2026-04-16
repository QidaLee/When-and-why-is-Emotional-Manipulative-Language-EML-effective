import functools
print = functools.partial(print, flush=True)

import torch
import pandas as pd
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

INPUT_DATA_PATH = "./data/Persuasion_For_Good/all_turns_data.xlsx"
OUTPUT_DATA_PATH = "./output_data/all_turns_data_with_eml_label.xlsx"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_MODEL_PATH = "./models/llama3_lora_manipulative"

EML = """
    * 'Minimization' = Language that invalidates a person's feelings, opinions, or emotional experience. Invalidating means to consider something weak, unaccepted, disrespected, or ineffective.
    * 'Power' Language that asserts dominance to control or intimidate others, exploiting elements like veiled threats, hierarchy, or authority.
    * 'Guilt' Language that blames a person to make them feel responsible or feel bad about some wrongdoing. It may contain accusations against others to state that they are at fault.
    * 'Shame' Language that to make others feel inferior, unworthy, or embarrassed including judgments, sarcasm, criticism, or put-downs. 
    """

examples_turn = """
    Input: "you couldn't even find a bone in your body to apologize for your part in this and to even reach out and tell her that you love her."
    Output: 1
    Explanation: The speaker is influencing the receiver to apologise by inducing negative emotions, as well as using guilt.

    Input: "I don't care okay I don't care whether you're tired I have an expectation I expect you both to do the homework."
    Output: 1
    Explanation: The text contains emotional manipulative language since the speaker is asking the receiver to do the homework by using authority and minimizing his/her emotions.

    Input: "No? Oh, we don't hit and then say sorry. Bronson, you need to come here. You're past your boundary. No, that is not acceptable. Ever."
    Output: 0
    Explanation: Sentences contains assertive language, and authority is used to influence but not to make the receiver feel bad.

    Input: "I feel like a prisoner in my own home. My children have been spanked, grounded, and we've had things taken away from them."
    Output: 0
    Explanation: There are no explicit attempts to influence or to achieve compliance.

    Input: "Brandon, look at me. Stop. No. You do not yell at mommy like that. Stand up. We're not doing hugs until you listen. You don't yell at mommy."
    Output: 1
    Explanation: There is emotionally manipulative language since the mum is emotionally punishing the child to make him/her stop yelling.

    Input: "You're past your boundary. No, that is not acceptable. Ever. You're going to the penalty box. You should feel ashamed."
    Output: 1
    Explanation: Emotional appeals (shame) is used to influence the receiver.

    Input: "I love you Dave. Where's your answer? Fuck off"
    Output: 0
    Explanation: Toxic language and hate speech is present, but no exploitation of emotional appeals to achieve compliance.
    """

def generate_prompt():
    return """
DEFINITION:
A text contains emotionally manipulative language if at least one of the following conditions is true:
1. The text is used to influence someone in doing or thinking something, not relying on reasoned argument but on the exploitation of emotional appeals to achieve compliance.
2. The text includes at least one of the following language techniques: """ + EML + """

EXAMPLES:
""" + examples_turn + """

TASK:
You will see a full conversation as context.
Judge only the current utterance.
Classify whether the current utterance is emotionally manipulative (1) or not (0).
Return only a single digit.

INPUT: """.strip()

instruction = generate_prompt()

def make_prompt(full_dialogue, current_utterance):
    return """
""" + instruction + """

FULL CONVERSATION CONTEXT:
""" + full_dialogue + """

CURRENT UTTERANCE TO JUDGE:
""" + current_utterance + """

ANSWER:
""".strip()

def load_trained_model():
    print("Loading Llama3 model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="cuda:0",
        token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return text

def predict_single(full_dialogue, current_utterance, tokenizer, model):
    prompt = make_prompt(full_dialogue, current_utterance)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_length=None
        )

    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    digits = re.findall(r'[01]', generated)
    label = int(digits[0]) if digits else 0
    return label, generated

def label_eml_data(tokenizer, model):
    print(f"\nLoading data from: {INPUT_DATA_PATH}")
    df = pd.read_excel(INPUT_DATA_PATH, dtype=str)
    print(f"Loaded {len(df)} rows")

    if "Sentence" not in df.columns:
        raise ValueError("Missing 'Sentence' column in dataset")

    # FIXED: Use exact column name Dialogue_ID
    dialog_id_col = "Dialogue_ID"
    if dialog_id_col not in df.columns:
        raise ValueError(f"Missing {dialog_id_col} column")

    df["clean_sentence"] = df["Sentence"].apply(clean_text)
    df = df.sort_values([dialog_id_col])

    eml_labels = []
    raw_outputs = []
    total = len(df)

    for dialog_id, group in df.groupby(dialog_id_col):
        group = group.reset_index(drop=True)
        utterances = group["clean_sentence"].tolist()
        full_dialogue = "\n".join([f"{u}" for u in utterances if u])

        for idx_in_dialog, row in group.iterrows():
            current_utt = row["clean_sentence"]
            if len(current_utt) < 3:
                eml_labels.append(0)
                raw_outputs.append("")
                continue

            label, raw = predict_single(full_dialogue, current_utt, tokenizer, model)
            eml_labels.append(label)
            raw_outputs.append(raw)

            global_idx = len(eml_labels) - 1
            if (global_idx + 1) % 100 == 0:
                print(f"Processed: {global_idx + 1}/{total} | Dialog ID: {dialog_id}")

    df["eml_label"] = eml_labels
    df["eml_raw_output"] = raw_outputs
    df = df.drop(columns=["clean_sentence"])

    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    df.to_excel(OUTPUT_DATA_PATH, index=False)
    print(f"\nCompleted! Results saved to: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    tokenizer, model = load_trained_model()
    label_eml_data(tokenizer, model)