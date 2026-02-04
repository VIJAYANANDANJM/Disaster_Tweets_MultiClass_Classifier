import torch

def explain_prediction(model, tokenizer, text, target_class=None, max_len=128):
    model.eval()

    # Tokenize
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Enable gradient tracking
    input_ids = input_ids.clone().detach()
    input_ids.requires_grad = False

    # Forward pass
    outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    logits = model.classifier(cls_embedding)

    probs = torch.softmax(logits, dim=1)

    if target_class is None:
        target_class = torch.argmax(probs, dim=1)

    # Backward pass for explanation
    model.zero_grad()
    logits[0, target_class].backward()

    # Token embeddings
    token_embeddings = outputs.last_hidden_state[0]

    # Gradient × Input
    token_importance = torch.norm(token_embeddings, dim=1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    explanation = list(zip(tokens, token_importance.detach().cpu().numpy()))

    return explanation, probs.detach().cpu().numpy()
