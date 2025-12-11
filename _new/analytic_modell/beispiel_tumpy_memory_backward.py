import torch
import torch.nn as nn



def _get_tensors_saved_for_backward(net: nn.Module, loss: torch.Tensor, duplicate=True) -> torch.Tensor:
    """
    Berechnet die Speicherkosten für den Backward-Pass eines Netzwerkes (nn.Module)
    anhand des Berechnungsgraphen der Verlustfunktion (loss).

    Mithilfe eines Breadth-First-Search (BFS) Traversals durch den Berechnungsgraphen
    werden alle Tensoren gesammelt, die für die Gradientenberechnung benötigt werden.
    Jede Berechnungsschritt (grad_fn) ist dabe ein Knoten im Graphen.
    Dieser kann über ein Attribute erreicht werden. Inoffiziell ist das über "_saved..." möglich.
    (Beispiel für ein Attribute ist: "_saved_mat")
    Dabei werden nur die Tensoren berücksichtigt, die zu den Parametern oder Buffern
    des Netzwerks gehören. Die Speicherkosten werden dann summiert.

    :param net:
    :param loss:
    :param duplicate:
    :return:
    """
    params = set(net.parameters())
    params = params.union(set(net.buffers()))
    saved_tensors = dict()  # saved_tensor --> operator name
    queue = [loss.grad_fn]
    visited_grad_fn = set()
    while queue:
        new_queue = []
        for grad_fn in queue:
            if grad_fn in visited_grad_fn:
                continue
            visited_grad_fn.add(grad_fn)
            names = [k for k in dir(grad_fn) if k.startswith('_saved')]  # names of saved tensors
            for k in names:
                v = getattr(grad_fn, k)
                if isinstance(v, torch.Tensor) and v not in params and v not in saved_tensors:
                    saved_tensors[v] = grad_fn.name()
            new_queue += [next_function for next_function, _ in grad_fn.next_functions if
                          next_function is not None and next_function not in visited_grad_fn]
        queue = new_queue

    if not duplicate:
        return saved_tensors

    # remove duplicated tensors created by views/slicing/in-place operations
    deduplicated = dict()  # ptr --> [tensor, operator_name]
    for tensor, operator_name in saved_tensors.items():
        ptr = tensor.storage().data_ptr()
        if ptr in deduplicated:
            value, old_operator_name = deduplicated[ptr]
            if value.numel() < tensor.numel():
                deduplicated[ptr] = [tensor, operator_name]
        else:
            deduplicated[ptr] = [tensor, operator_name]

    return deduplicated

def memory_for_backward(net: nn.Module, loss: torch.Tensor) -> int:
    deduplicated = _get_tensors_saved_for_backward(net, loss)
    saved_memory = sum([tensor.element_size() * tensor.numel() for ptr, [tensor, operator_name] in deduplicated.items()])
    return saved_memory

def run_test():
    model = nn.Linear(in_features=1000 ,out_features=10 , bias=False)
    input_tensor = torch.randn(1, 1000, requires_grad=True)

    output = model(input_tensor)
    loss = output.sum()
    mem_bytes = memory_for_backward(model, loss)

    num_elements = 1* 1000 # Batch * features
    bytes_per_float = 4 # float32 -> 4 bytes
    expected_bytes = num_elements * bytes_per_float


    if mem_bytes == expected_bytes:
        print("\n✅ TEST ERFOLGREICH: Messung stimmt exakt mit der Theorie überein!")
    else:
        print("\n❌ TEST FEHLGESCHLAGEN: Abweichung gefunden.")

        # Zusatz-Info: Welche Tensoren wurden gefunden?
    tensors = _get_tensors_saved_for_backward(model, loss)
    print("\nGefundene Tensoren Details:")
    for ptr, (tens, op_name) in tensors.items():
        print(f" - Op: {op_name}, Shape: {list(tens.shape)}, Größe: {tens.numel() * tens.element_size()} Bytes")

if __name__ == "__main__":
    run_test()


