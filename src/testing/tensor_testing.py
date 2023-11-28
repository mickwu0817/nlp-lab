import torch


def show_tensor(x: torch.Tensor):
    # print(f"{type(x)} : {x.shape} : {x}")
    print(f"{type(x)} : {x.shape}")


def try1():
    show_tensor(torch.tensor(1))
    show_tensor(torch.tensor([1]))
    show_tensor(torch.tensor([[1]]))
    show_tensor(torch.tensor([[[1]]]))
    show_tensor(torch.tensor([[[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]],[[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]]]))
    # show_tensor(torch.tensor([[[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]],[[[[1]]]]]))


def demo_shap_dtype():
    t1 = torch.tensor([1, 2, 3])
    print(f"{t1} : {t1.shape} : {t1.dtype}")


def demo_cat():
    t1 = torch.tensor([[1, 2, 3]])
    t2 = torch.tensor([[4, 5, 6]])
    t = torch.cat((t1, t2), 0)
    print(f"{t1.shape} : {t1}")
    print(f"{t2.shape} : {t2}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_stack():
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    t = torch.stack((t1, t2), 0)
    print(f"{t1.shape} : {t1}")
    print(f"{t2.shape} : {t2}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_chunk():
    t1 = torch.tensor([1, 2, 3, 4, 5, 6])
    t = torch.chunk(t1, 3, 0)
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t}")


def demo_reshape():
    t1 = torch.tensor([1, 2, 3, 4, 5, 6])
    t = torch.reshape(t1, (2, 3))
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_transpose():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t = torch.transpose(t1, 0, 1)
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_sequeeze():
    t1 = torch.tensor([[[1], [2], [3], [4], [5], [6]]])
    t = torch.squeeze(t1)
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_unsequeeze():
    t1 = torch.tensor([[[1], [2], [3], [4], [5], [6]]])
    t = torch.unsqueeze(t1, 2)
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_expand():
    t1 = torch.tensor([[1, 2, 3, 4, 5, 6]])
    t = t1.expand(2, 6)
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t.shape} : {t}")


def demo_repeat():
    t1 = torch.tensor([[1, 2, 3, 4, 5, 6]])
    t = t1.repeat(2, 1)
    print(f"{t1.shape} : {t1}")
    print(f"===========")
    print(f"{t.shape} : {t}")


if __name__ == '__main__':
    # demo_shap_dtype()
    demo_cat()
    # demo_stack()
    # demo_chunk()
    # demo_reshape()
    # demo_transpose()
    # demo_sequeeze()
    # demo_unsequeeze()
    # demo_expand()
    # demo_repeat()

    pass


