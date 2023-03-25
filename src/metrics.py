# Author: Navaneethan Vaikunthan, Atharva Naik
# common metrics for the task.

# classification based setup metrics:
def get_class_preds(outputs, id2label):
    preds = outputs.logits.argmax(dim=-1)
    pred_words = [id2label[pred.cpu().item()] for pred in preds]
    return pred_words

def proxy_accuracy(inputs, generations, tokenizer):
    decoded_preds = tokenizer.batch_decode(generations, skip_special_tokens=True)
    answers = inputs["answer"]
    for i, answer in enumerate(answers):
        print(f"Question: {inputs['question'][i]}")
        print(f"Prediction: {decoded_preds[i]}")
        print(f"Answer: {answer}")

def class_accuracy(inputs, outputs, id2label):

    pred_words = get_class_preds(outputs, id2label)
    print(pred_words)
    print(inputs["answer"])
    # print((pred_words == inputs["answer"]).sum() / len(pred_words))

# generative setting metrics:
