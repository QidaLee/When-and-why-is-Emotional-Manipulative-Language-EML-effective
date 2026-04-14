import functools
print = functools.partial(print, flush=True)

import torch
import pandas as pd
import os
import re
import string
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

HF_TOKEN = "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"
INPUT_DATA_PATH = "./data/Persuasion_For_Good/all_turns_data.xlsx"
OUTPUT_DATA_PATH = "./output_data/all_turns_data_with_eml_label.xlsx"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_MODEL_PATH = "./models/llama3_lora_manipulative"
CONFIDENCE_THRESHOLD = 0.6

# ===== 训练时完全一样的 prompt =====
EML = """
    * 'Minimization' = Language that invalidates a person's feelings, opinions, or emotional experience. Invalidating means to consider something weak, unaccepted, disrespected, or ineffective.
    * 'Power'. Language that asserts dominance to control or intimidate others, exploiting elements like veiled threats, hierarchy, or authority.
    * 'Guilt'. Language that blames a person to make them feel responsible or feel bad about some wrongdoing. It may contain accusations against others to state that they are at fault.
    * 'Shame'. Language that to make others feel inferior, unworthy, or embarrassed (e.g., including judgments, sarcasm, criticism, or put-downs). 
    """

examples_new = f"""
    Input: ["I love you Dave" "Where's your answer?" "you couldn't even find a bone in your body to apologize for your part in this and to even reach out and tell her that you love her."]
    Output: 1
    Explanation: In the last sentence, the speaker is influencing the receiver to apologise by inducing negative emotions, as well as using guilt.

    Input: ["I want to go play" "Leave them alone babe" "I don't care okay I don't care whether you're tired I have an expectation I expect you both to do the homework" "Mum" "Please I said please"]
    Output: 1
    Explanation: In this conversation there is a sentence containing emotional manipulative language since the speaker is asking the receiver to do the homework by using authority and minimizing his/her emotions (i.e. not caring if he/she feels tired).

    Input: ["No!" "Michael, we can speak to Sean after he's at a timeout, okay?" "She wanna acts a little silly." "Sean, you do need to tell me that you're sorry." "I'm waiting."]
    Output: 1
    Explanation: In this conversation there are attempts to influence (ask for apologies) and shame (being silly).

    Input: ["No? Oh, we don't hit and then say sorry. Bronson, you need to come here." "You're past your boundary." "No, that is not acceptable. Ever. You're going to the penalty box."]
    Output: 0
    Explanation: Sentences contains assertive language, and authority is used to influence but not to make the receiver feel bad.

    Input: ["I feel like a prisoner in my own home. My children have been spanked, grounded, and we've had things taken away from them."]
    Output: 0
    Explanation: There are no explicit attempts to influence or to achieve compliance.

    Input: ["Brandon, look at me." "Stop." "No. Stop." "You do not yell at mommy like that." "Stand up. We're not doing hugs until you listen. You don't yell at mommy."]
    Output: 1
    Explanation: There is emotionally manipulative language since the mum is emotionally punishing the child (not giving hugs) to make him/her stop yelling.

    Input: ["You're past your boundary." "No, that is not acceptable. Ever. You're going to the penalty box. You should feel ashamed."]
    Output: 1
    Explanation: In the last sentence, emotional appeals (shame) is used to influence the receiver (to come there or, probably, to stop some kind of behavior).

    Input: ["I love you Dave" "Where's your answer?" "Fuck off"]
    Output: 0
    Explanation: Toxic language and hate speech is present, but no exploitation of emotional appeals to achieve compliance.
    """


def generate_prompt():
    return f"""
        DEFINITION 1: 
        A text contains emotionally manipulative language if at least one the following conditions are true:
        * The text is used to influence someone else in doing or thinking something, not relying on reasoned argument but on the exploitation of emotional appeals to achieve compliance.
        * The text includes at least one of the following language techniques of manipulation: 
            {EML}


        EXAMPLES: 
        {examples_new}


        TASK: 
        Classify whether the conversation contains emotionally manipulative language (1) or not (0). Return a single digit.


        INPUT: """.strip()


instruction = generate_prompt()


def make_prompt(utterance):
    return f"{instruction}{utterance}\n\nANSWER:"


# ===== 模型加载 =====
def load_trained_model():
    print("🔗 加载 Llama3 模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token = HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map = "cuda:0",
        token = HF_TOKEN
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    print("✅ 模型加载成功！")
    return tokenizer, model


# ===== 清理文本 =====
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    return str(text).strip()


# ===== 单条推理 =====
def predict_single(text, tokenizer, model):
    prompt = make_prompt(text)
    inputs = tokenizer(prompt, return_tensors = "pt", truncation = True, max_length = 1024)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 3,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id,
            max_length = None
        )

    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens = True
    ).strip()

    label = 1 if "1" in generated else 0
    return label, generated


# ===== 批量标注 =====
def label_eml_data(tokenizer, model):
    print(f"\n加载数据: {INPUT_DATA_PATH}")
    df = pd.read_excel(INPUT_DATA_PATH, dtype = str)
    print(f"✅ 成功加载 {len(df)} 条数据")

    if "Sentence" not in df.columns:
        raise ValueError("❌ 数据缺少 'Sentence' 列")

    df["clean_sentence"] = df["Sentence"].apply(clean_text)
    text_list = df["clean_sentence"].tolist()

    eml_labels = []
    raw_outputs = []
    total = len(text_list)

    for idx, text in enumerate(text_list):
        if len(text) < 5:
            eml_labels.append(0)
            raw_outputs.append("")
            continue

        label, raw = predict_single(text, tokenizer, model)
        eml_labels.append(label)
        raw_outputs.append(raw)

        if (idx + 1) % 100 == 0:
            print(f"已标注: {idx + 1}/{total}")

    df["eml_label"] = eml_labels
    df["eml_raw_output"] = raw_outputs  # 保留原始输出方便debug
    df = df.drop(columns = ["clean_sentence"])

    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok = True)
    df.to_excel(OUTPUT_DATA_PATH, index = False)
    print(f"\n🎉 完成！结果保存到: {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    tokenizer, model = load_trained_model()
    label_eml_data(tokenizer, model)
