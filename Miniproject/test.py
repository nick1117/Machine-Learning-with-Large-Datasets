# reload full model
model = ConvNet().to(device)
model.load_state_dict(torch.load('original_model.pt'))
print("Loaded original model weights.")

activation_storage = {}

def get_activation_hook(name):
    def hook(model, input, output):
        mean_activations = output.detach().mean(dim=(2, 3))  # [batch, channels]
        if name not in activation_storage:
            activation_storage[name] = []
        activation_storage[name].append(mean_activations.cpu())
    return hook

hooks = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        hooks.append(module.register_forward_hook(get_activation_hook(name)))

model.eval()
with torch.no_grad():
    for i, (inputs, _) in enumerate(train_loader):
        if i >= 200:  # subset of training data
            break
        inputs = inputs.to(device)
        _ = model(inputs)

avg_activations = {}
for layer_name, activations in activation_storage.items():
    all_acts = torch.cat(activations, dim=0)  # [num_samples, channels]
    avg_per_filter = all_acts.mean(dim=0)     # [channels]
    avg_activations[layer_name] = avg_per_filter

# recommended to clean up hooks
for h in hooks:
    h.remove()

k_percent = 0.05 # prune bottom X%
pruned_indices = {}

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and name in avg_activations:
        avg = avg_activations[name]
        num_filters = avg.shape[0]
        num_prune = int(k_percent * num_filters)
        prune_idx = torch.argsort(avg)[:num_prune]
        pruned_indices[name] = prune_idx

        with torch.no_grad():
            module.weight[prune_idx] = 0
            if module.bias is not None:
                module.bias[prune_idx] = 0

print("Pruning complete")

weight_masks = {}

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        mask = torch.ones_like(module.weight)
        if name in pruned_indices:
            idx = pruned_indices[name]
            mask[idx] = 0
        weight_masks[name] = mask
    
def train_one_epoch_with_mask(model, train_loader, optimizer, criterion, device, weight_masks):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Fine-Tuning", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and name in weight_masks:
                    module.weight.grad *= weight_masks[name]
                    if module.bias is not None and module.bias.grad is not None:
                        module.bias.grad *= weight_masks[name][:, 0, 0, 0]

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    return train_loss, train_acc

val_loss, val_accuracy = validate(model, val_loader, criterion, device)
print(f"Post-Pruning Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

# fine-tuning epochs
for epoch in range(8):
    train_loss, train_accuracy = train_one_epoch_with_mask(model, train_loader, optimizer, criterion, device, weight_masks)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    print(f'[Fine-Tuning Epoch {epoch+1}] Val Accuracy: {val_accuracy:.2f}%')


val_loss, val_accuracy = validate(model, val_loader, criterion, device)
print(f"Post-Pruning AND post-fine tuning Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

def count_zeroed_and_total_weights(model):
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    print(f"Total weights     : {total_params:,}")
    print(f"Zeroed weights    : {zero_params:,}")
    print(f"Percentage pruned : {100 * zero_params / total_params:.2f}%")
    return total_params, zero_params

total_params, zero_params = count_zeroed_and_total_weights(model)


print(f"k percent: {k_percent}")
print(f"Final Validation Accuracy: {val_accuracy}")
#score = (accuracy + num zero weights / total parameters) / 2

score = (val_accuracy/100 + zero_params / total_params ) / 2
print("Score must be .36 or higher")
print(f"Final score for assignment: {score}")