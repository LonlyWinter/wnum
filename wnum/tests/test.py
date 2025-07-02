# %%
import numpy as np
import time

def arr_to_vec_single(a: list, dig: int) -> str:
    res = list()
    for i in a:
        if isinstance(i, list):
            res.append(arr_to_vec_single(i, dig))
        else:
            if dig == 1:
                res.append("{:.1f}".format(i))
            elif dig == 2:
                res.append("{:.2f}".format(i))
            elif dig == 3:
                res.append("{:.3f}".format(i))
            elif dig == 4:
                res.append("{:.4f}".format(i))
            elif dig == 5:
                res.append("{:.5f}".format(i))
            elif dig == 6:
                res.append("{:.6f}".format(i))
            elif dig == 7:
                res.append("{:.7f}".format(i))
            elif dig == 8:
                res.append("{:.8f}".format(i))
            else:
                res.append(str(i))
    return "vec![{}]".format(", ".join(res))

def print_arr(arr, dig: int = 0):
    a: list = arr.tolist()
    b = arr_to_vec_single(a, dig)
    print(arr.shape)
    print(b)

# %%
def arr2():
    a = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]]
    b = [[10.0, 20.0, 30.0, 40.0, 50.0],
        [60.0, 70.0, 80.0, 90.0, 100.0],
        [110.0, 120.0, 130.0, 140.0, 150.0]]
    a = np.asarray(a)
    b = np.asarray(b)
    start = time.time()
    for _ in range(1000000):
        np.matmul(a, b)
    end = time.time()
    print(end - start)

# arr2()
# %%
def arr3():
    a = np.arange(0, 120).reshape((2, 4, 3, 5))
    b = np.arange(200, 320).reshape((2, 4, 5, 3))
    start = time.time()
    for _ in range(1000000):
        np.matmul(a, b)
    end = time.time()
    print(end - start)
    print(np.matmul(a, b).shape, a, b, np.matmul(a, b))

# arr3()
# %%
def arr4_t():
    a = np.arange(0, 240).reshape((8, 2, 3, 5))
    b = a.transpose(1, 0, 2, 3)
    print_arr(b)

# arr4_t()
# %%
def arr4_concat():
    a = np.arange(0, 120).reshape((2, 4, 3, 5))
    b = np.arange(200, 360).reshape((2, 4, 4, 5))
    c = np.concatenate([a, b], 2)
    print_arr(c)

# arr4_concat()
# %%
def arr4_concat2():
    a = np.arange(0, 120).reshape((2, 4, 3, 5))
    b = np.arange(200, 360).reshape((2, 4, 4, 5))
    aa = a.transpose(2, 1, 0, 3)
    bb = b.transpose(2, 1, 0, 3)
    cc = np.concatenate([aa, bb], 0)
    dd = cc.transpose(2, 1, 0, 3)
    print_arr(dd)

# arr4_concat2()
# %%
def arr4_stack():
    a = np.arange(0, 120).reshape((2, 4, 3, 5))
    b = np.arange(200, 320).reshape((2, 4, 3, 5))
    c = np.stack([a, b], 2)
    print_arr(c)

# arr4_stack()
# %%
def arr5_matmul():
    a = np.arange(0, 12).reshape((2, 2, 3))
    b = np.arange(0, 12).reshape((2, 3, 2))
    c = np.matmul(a, b)
    print_arr(c)

# arr5_matmul()
# %%
def arr5_sum():
    a = np.arange(0, 120).reshape((2, 4, 3, 5))
    c = np.sum(a, 2, keepdims=True)
    print_arr(c)

# arr5_sum()
# %%
def arr5_broadcast():
    a = np.arange(0, 40).reshape((2, 4, 1, 5))
    c = np.broadcast_to(a, (2, 4, 3, 5))
    print_arr(c)

# arr5_broadcast()
# %%
def arr5_exp():
    a = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]]
    a = np.asarray(a)
    c = np.exp(a)
    print_arr(c, 2)

# arr5_exp()
# %%
def arr5_softmax():
    a = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]]
    a = np.asarray(a)
    m = np.max(a, 1, keepdims=True)
    d = a - m
    n = np.exp(d)
    den = np.sum(n, 1, keepdims=True)
    c = n/den
    print_arr(c, 2)

# arr5_softmax()
# %%
def arr5_broadcast1():
    a = np.arange(0, 40).reshape((2, 4, 5))
    c = np.concat([a, a], 1)
    print_arr(c)

# arr5_broadcast1()
# %%
def tensor_linear_grad():
    import torch
    from torch import optim

    w_gen = torch.tensor([[3.], [1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimizer = optim.SGD(m.parameters(), lr=0.001)
    for step in range(10):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss0 = (ys - sample_ys)**2
        loss = loss0.sum()
        loss.backward()
        optimizer.step()
        print(step, "ys", ys.tolist())
        print(step, "m.weight", m.weight.tolist())
        print(step, "m.bias", m.bias.tolist())
        print(step, "loss0", loss0.tolist())
        print(step, "loss", loss.tolist())

# tensor_linear_grad()
# %%
def tensor_linear_grad2():
    import torch
    from torch import optim
    from torch.nn import MSELoss

    w_gen = torch.tensor([[3.], [1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen) + b_gen

    criterion = MSELoss()

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimizer = optim.SGD(m.parameters(), lr=0.001)
    for step in range(10):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss = criterion(ys, sample_ys)
        loss.backward()
        optimizer.step()
        print(step, "ys", ys.tolist())
        print(step, "m.weight", m.weight.tolist())
        print(step, "m.bias", m.bias.tolist())
        print(step, "loss", loss.tolist())

# tensor_linear_grad2()
# %%
# %%
def arr6_matmul():
    a = np.arange(0, 4).reshape((4, 1))
    b = np.arange(0, 4).reshape((1, 4))
    c = np.matmul(a, b)
    print_arr(a)
    print_arr(b)
    print_arr(c)

# arr6_matmul()
# %%
def arr7_matmul():
    a = np.arange(0, 40).reshape((5, 8))
    b = np.arange(0, 56).reshape((8, 7))
    c = np.matmul(a, b)
    print_arr(a, 2)
    print_arr(b, 2)
    print_arr(c, 2)

# arr7_matmul()


# %%
# %%
def arr4_transpose5():
    a = np.arange(0, 7840).reshape((784, 10))
    b = a.T
    print_arr(a, 2)
    print_arr(b, 2)

# arr4_transpose5()


# %%
def conv2d_1():
    import torch
    
    t = torch.randn(1, 4, 5, 5).round(decimals=1)
    w = torch.randn(2, 4, 3, 3).round(decimals=1)

    # t = torch.from_numpy(np.asarray([0.85964072, 0.71211761, -1.09406805, -0.77558726, -0.02582738, 3.97113061, -1.10437846, -1.70252490, -0.05771952, 0.27803150, -0.39693168, -0.46601918, 0.53663176, 1.42919457, -0.85962903, -0.77167690, 0.05770938, 0.65752822, 0.71924329, 1.09647822, 1.51268589, 0.57108110, -0.28455755, -0.16131642, -2.17233372, 0.69238836, -1.88622403, 0.06579538, 1.22393274, 1.33153725, 0.47411421, -0.14155859, -1.12750423, 1.46705627, -0.26525563, -1.04082060, -1.65568209, -0.70816612, 0.10296751, 1.04969800, 0.07992742, -0.00734435, -2.05525494, -1.07068956, -0.63715690, -1.26226807, 0.08093969, 1.09288728, 0.59042156, 0.40313730, -1.30234063, 1.53539503, -0.30652791, -0.22806698, -0.72191185, -1.10219038, 0.53017122, -0.16020557, -2.75541496, 0.65775508, -0.70478845, 0.18683709, 0.76295644, -2.05025673, -0.52546084, -0.20815697, -2.48268986, 0.18845695, -0.31933388, -1.72322977, 0.68082523, -0.33391404, 0.45952773, -0.36922431, 0.48055553, -0.16143925, 0.03446232, -0.64370668, 0.85135740, -0.26065645, 0.24666984, -0.27955797, 0.17713474, -0.73822081, -0.33799267, -0.33449483, -0.19884871, 0.72386813, -0.22002393, -0.47659349, -0.49052975, -0.08234675, 0.31365600, -1.53247166, -0.65478599, -0.74459970, 1.09684443, -0.29384729, 1.93104136, 0.77600366])
    #                      .reshape((1, 4, 5, 5)))
    # w = torch.from_numpy(np.asarray([0.98441446, 1.13562489, -1.75485373, 0.23127152, -0.61426204, -0.67533791, 0.49990097, -2.18303776, -0.56340230, -0.55286753, -0.59479386, 0.83068454, -1.34519529, -0.10893888, -0.19163457, -0.70227253, -0.45601600, -1.23771870, -0.08224025, -0.92343986, -0.08192176, -1.07030106, 0.62783027, 0.34265703, -0.00711847, -2.01374531, -0.59634608, 0.55641413, -0.17700839, -0.03821314, -0.15610510, -0.05534480, 0.43968496, 0.03338196, -1.24181759, -0.75808424, 0.77903545, -1.24383652, 0.49347490, -1.56421483, 0.10321048, -1.63740206, -0.21376657, 0.32510963, 1.00355470, -0.02697560, 0.22431640, 0.23358896, -0.97666478, -0.80173451, -0.87425184, 0.98812670, 1.20250547, 0.79542422, 0.39937261, -1.32567835, 0.95923585, 1.54405212, 0.10927629, -0.07796084, 1.43363726, 0.99500340, -3.03855228, -0.28374413, -0.35775465, 0.52509534, 1.41767383, -0.09767344, -1.67028511, -1.11630857, -0.01456974, -2.10364056])
    #                      .reshape((2, 4, 3, 3)))
    
    print_arr(t.flatten(), 2)
    print_arr(w.flatten(), 2)

    res = torch.nn.functional.conv2d(t, w, None, 1, 0, 1, 1).round(decimals=2)
    print_arr(res.flatten())
    
    print(t.shape, w.shape)
    res = torch.nn.functional.conv2d(t, w, None, 2, 2, 2, 1).round(decimals=2)
    print_arr(res.flatten())

# conv2d_1()
# %%
# %%
def conv2d_2():
    import torch
    # dim self: batch_size, channels_in_data, data_h, data_w
    # dim kernel: channels_out, channels_in, data_h, data_w
    t = torch.randn(1, 6, 5, 5).round(decimals=1)
    w = torch.randn(3, 2, 3, 3).round(decimals=1)

    # t = torch.from_numpy(np.asarray([0.85964072, 0.71211761, -1.09406805, -0.77558726, -0.02582738, 3.97113061, -1.10437846, -1.70252490, -0.05771952, 0.27803150, -0.39693168, -0.46601918, 0.53663176, 1.42919457, -0.85962903, -0.77167690, 0.05770938, 0.65752822, 0.71924329, 1.09647822, 1.51268589, 0.57108110, -0.28455755, -0.16131642, -2.17233372, 0.69238836, -1.88622403, 0.06579538, 1.22393274, 1.33153725, 0.47411421, -0.14155859, -1.12750423, 1.46705627, -0.26525563, -1.04082060, -1.65568209, -0.70816612, 0.10296751, 1.04969800, 0.07992742, -0.00734435, -2.05525494, -1.07068956, -0.63715690, -1.26226807, 0.08093969, 1.09288728, 0.59042156, 0.40313730, -1.30234063, 1.53539503, -0.30652791, -0.22806698, -0.72191185, -1.10219038, 0.53017122, -0.16020557, -2.75541496, 0.65775508, -0.70478845, 0.18683709, 0.76295644, -2.05025673, -0.52546084, -0.20815697, -2.48268986, 0.18845695, -0.31933388, -1.72322977, 0.68082523, -0.33391404, 0.45952773, -0.36922431, 0.48055553, -0.16143925, 0.03446232, -0.64370668, 0.85135740, -0.26065645, 0.24666984, -0.27955797, 0.17713474, -0.73822081, -0.33799267, -0.33449483, -0.19884871, 0.72386813, -0.22002393, -0.47659349, -0.49052975, -0.08234675, 0.31365600, -1.53247166, -0.65478599, -0.74459970, 1.09684443, -0.29384729, 1.93104136, 0.77600366])
    #                      .reshape((1, 4, 5, 5)))
    # w = torch.from_numpy(np.asarray([0.98441446, 1.13562489, -1.75485373, 0.23127152, -0.61426204, -0.67533791, 0.49990097, -2.18303776, -0.56340230, -0.55286753, -0.59479386, 0.83068454, -1.34519529, -0.10893888, -0.19163457, -0.70227253, -0.45601600, -1.23771870, -0.08224025, -0.92343986, -0.08192176, -1.07030106, 0.62783027, 0.34265703, -0.00711847, -2.01374531, -0.59634608, 0.55641413, -0.17700839, -0.03821314, -0.15610510, -0.05534480, 0.43968496, 0.03338196, -1.24181759, -0.75808424, 0.77903545, -1.24383652, 0.49347490, -1.56421483, 0.10321048, -1.63740206, -0.21376657, 0.32510963, 1.00355470, -0.02697560, 0.22431640, 0.23358896, -0.97666478, -0.80173451, -0.87425184, 0.98812670, 1.20250547, 0.79542422, 0.39937261, -1.32567835, 0.95923585, 1.54405212, 0.10927629, -0.07796084, 1.43363726, 0.99500340, -3.03855228, -0.28374413, -0.35775465, 0.52509534, 1.41767383, -0.09767344, -1.67028511, -1.11630857, -0.01456974, -2.10364056])
    #                      .reshape((2, 4, 3, 3)))
    
    print_arr(t.flatten(), 1)
    print_arr(w.flatten(), 1)

    print(t.shape, w.shape)
    res = torch.nn.functional.conv2d(t, w, None, 2, 2, 2, 3)
    print_arr(res.flatten(), 2)

# conv2d_2()

# %%
def conv2d_3():
    import torch
    
    t = torch.randn(3, 4, 7, 7).round(decimals=1)
    w = torch.randn(2, 4, 3, 3).round(decimals=1)

    # t = torch.from_numpy(np.asarray([0.85964072, 0.71211761, -1.09406805, -0.77558726, -0.02582738, 3.97113061, -1.10437846, -1.70252490, -0.05771952, 0.27803150, -0.39693168, -0.46601918, 0.53663176, 1.42919457, -0.85962903, -0.77167690, 0.05770938, 0.65752822, 0.71924329, 1.09647822, 1.51268589, 0.57108110, -0.28455755, -0.16131642, -2.17233372, 0.69238836, -1.88622403, 0.06579538, 1.22393274, 1.33153725, 0.47411421, -0.14155859, -1.12750423, 1.46705627, -0.26525563, -1.04082060, -1.65568209, -0.70816612, 0.10296751, 1.04969800, 0.07992742, -0.00734435, -2.05525494, -1.07068956, -0.63715690, -1.26226807, 0.08093969, 1.09288728, 0.59042156, 0.40313730, -1.30234063, 1.53539503, -0.30652791, -0.22806698, -0.72191185, -1.10219038, 0.53017122, -0.16020557, -2.75541496, 0.65775508, -0.70478845, 0.18683709, 0.76295644, -2.05025673, -0.52546084, -0.20815697, -2.48268986, 0.18845695, -0.31933388, -1.72322977, 0.68082523, -0.33391404, 0.45952773, -0.36922431, 0.48055553, -0.16143925, 0.03446232, -0.64370668, 0.85135740, -0.26065645, 0.24666984, -0.27955797, 0.17713474, -0.73822081, -0.33799267, -0.33449483, -0.19884871, 0.72386813, -0.22002393, -0.47659349, -0.49052975, -0.08234675, 0.31365600, -1.53247166, -0.65478599, -0.74459970, 1.09684443, -0.29384729, 1.93104136, 0.77600366])
    #                      .reshape((1, 4, 5, 5)))
    # w = torch.from_numpy(np.asarray([0.98441446, 1.13562489, -1.75485373, 0.23127152, -0.61426204, -0.67533791, 0.49990097, -2.18303776, -0.56340230, -0.55286753, -0.59479386, 0.83068454, -1.34519529, -0.10893888, -0.19163457, -0.70227253, -0.45601600, -1.23771870, -0.08224025, -0.92343986, -0.08192176, -1.07030106, 0.62783027, 0.34265703, -0.00711847, -2.01374531, -0.59634608, 0.55641413, -0.17700839, -0.03821314, -0.15610510, -0.05534480, 0.43968496, 0.03338196, -1.24181759, -0.75808424, 0.77903545, -1.24383652, 0.49347490, -1.56421483, 0.10321048, -1.63740206, -0.21376657, 0.32510963, 1.00355470, -0.02697560, 0.22431640, 0.23358896, -0.97666478, -0.80173451, -0.87425184, 0.98812670, 1.20250547, 0.79542422, 0.39937261, -1.32567835, 0.95923585, 1.54405212, 0.10927629, -0.07796084, 1.43363726, 0.99500340, -3.03855228, -0.28374413, -0.35775465, 0.52509534, 1.41767383, -0.09767344, -1.67028511, -1.11630857, -0.01456974, -2.10364056])
    #                      .reshape((2, 4, 3, 3)))
    
    print_arr(t.flatten(), 2)
    print_arr(w.flatten(), 2)

    res = torch.nn.functional.conv2d(t, w, None, 1, 0, 1, 1).round(decimals=2)
    print_arr(res.flatten(), 2)
    print(t.shape, w.shape, res.shape)
    
    res = torch.nn.functional.conv2d(t, w, None, 2, 3, 5, 1).round(decimals=2)
    print_arr(res.flatten(), 2)

    print(t.shape, w.shape, res.shape)

# conv2d_3()

# %%
def conv_transpose2d_1():
    import torch
    
    # t = torch.randn(3, 4, 3, 3).round(decimals=1)
    # w = torch.randn(2, 4, 2, 2).round(decimals=1)

    t = torch.from_numpy(np.asarray([
        0.4056, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
        1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
        1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
        0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
        1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
        0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
        0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
        0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
        -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
        -0.8, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
    ], dtype=np.float32).reshape((1, 4, 5, 5)))
    w = torch.from_numpy(np.asarray([
        -0.9325, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
        -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
        -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
        0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
        0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
        -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
        1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
        0.5583, 0.4623, 0.6026,
    ], dtype=np.float32).reshape((2, 4, 3, 3)))
    
    w = w.transpose(0, 1)
    res = torch.nn.functional.conv_transpose2d(t, w)
    print_arr(res.flatten(), 4)
    print(t.shape, w.shape, res.shape)

conv_transpose2d_1()
# %%
def gather_scatter_1():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    b = torch.from_numpy(np.asarray([
        [0, 2, 1],
        [1, 0, 2],
    ]))
    res = a.gather(dim=1, index=b)
    print(a, a.shape, b, b.shape, res, res.shape, sep="\n\n")
    print_arr(a.flatten())
    print_arr(b.flatten())
    print_arr(res.flatten())

    res_scatter = torch.zeros_like(a).scatter(dim=1, index=b, src=res)
    print(res_scatter, res_scatter.shape)

gather_scatter_1()
# %%
def gather_scatter_2():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    b = torch.from_numpy(np.asarray([
        [0, 1, 0],
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
    ]))
    res = a.gather(dim=0, index=b)
    print(a, a.shape, b, b.shape, res, res.shape, sep="\n\n")
    print_arr(a.flatten())
    print_arr(b.flatten())
    print_arr(res.flatten())

    res_scatter = torch.zeros_like(a).scatter(dim=0, index=b, src=res)
    print(res_scatter, res_scatter.shape)

gather_scatter_2()
# %%
def nll_and_cross_entropy():
    import torch
    a = torch.from_numpy(np.asarray([
        [ 1.1050,  0.3013, -1.5394, -2.1528, -0.8634],
        [ 1.0730, -0.9419, -0.1670, -0.6582,  0.5061],
        [ 0.8318,  1.1154, -0.3610,  0.5351,  1.0830]
    ]))
    b = torch.from_numpy(np.asarray([1, 0, 4]))
    res1 = torch.nn.functional.log_softmax(a, dim=1)
    print_arr(res1, dig=6)
    res2 = torch.nn.functional.nll_loss(res1, b)
    print(res2)
    res3 = torch.nn.functional.cross_entropy(a, b)
    print(res3)
nll_and_cross_entropy()
# %%
def grad_relu():
    import torch
    a = torch.from_numpy(np.arange(-6, 6, dtype=np.float32).reshape((2, 2, 3)))
    a.requires_grad = True
    b = a.relu()
    c = b.sum()
    c.backward()

    print_arr(b)
    print_arr(a.grad)
    print_arr(a)
grad_relu()
# %%
def grad_gather_1():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = torch.from_numpy(np.asarray([
        [0, 2, 1],
        [1, 0, 2],
    ]))
    res = a.gather(dim=1, index=b)
    print(a, a.shape, b, b.shape, res, res.shape, sep="\n\n")
    print_arr(a)
    print_arr(b)
    print_arr(res)
    loss = res.sum()
    loss.backward()
    print_arr(a.grad)
grad_gather_1()
# %%
def grad_gather_2():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = torch.from_numpy(np.asarray([
        [0, 2, 1],
        [1, 0, 2],
    ]))
    c = torch.from_numpy(np.arange(10, 16, dtype=np.float32).reshape((3, 2)))
    res = a.gather(dim=1, index=b)
    loss = res.matmul(c).mean()
    loss.backward()
    print(a, a.shape, b, b.shape, res, res.shape, sep="\n\n")
    print_arr(a)
    print_arr(b)
    print_arr(c)
    print(loss, loss.shape)
    print_arr(res)
    print_arr(a.grad)
grad_gather_2()

# %%
def grad_gather_3():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = torch.from_numpy(np.asarray([
        [0, 2, 1],
        [1, 0, 2],
    ]))
    c = torch.from_numpy(np.arange(0, 2).reshape((2, )))
    res = a.gather(dim=1, index=b)
    loss = torch.nn.functional.nll_loss(res, c)
    loss.backward()
    print(a, a.shape, b, b.shape, res, res.shape, sep="\n\n")
    print_arr(a)
    print_arr(b)
    print_arr(c)
    print(loss, loss.shape)
    print_arr(res)
    print_arr(a.grad)
grad_gather_3()
# %%
def grad_unsqueeze_logsofmax_1():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((6, )))
    a.requires_grad = True
    b = a.unsqueeze(0).log_softmax(1)
    c = torch.from_numpy(np.arange(3, 4).reshape((1, )))
    loss = torch.nn.functional.nll_loss(b, c)
    print(b, b.shape, c, c.shape, loss, sep="\n")
    loss.backward()
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c)
    print_arr(a.grad, dig=4)
grad_unsqueeze_logsofmax_1()
# %%
def grad_unsqueeze_sofmax_1():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((6, )))
    a.requires_grad = True
    b = a.unsqueeze(0).softmax(1)
    c = torch.from_numpy(np.arange(3, 4).reshape((1, )))
    loss = torch.nn.functional.nll_loss(b, c)
    print(b, b.shape, c, c.shape, loss, sep="\n")
    loss.backward()
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c)
    print_arr(a.grad, dig=4)
grad_unsqueeze_sofmax_1()
# %%
def grad_ln_1():
    import torch
    a = torch.from_numpy(np.arange(1, 7, dtype=np.float32).reshape((6, )))
    a.requires_grad = True
    b = a.log()
    loss = b.sum()
    print(b, b.shape, loss, sep="\n")
    loss.backward()
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(a.grad, dig=4)
grad_ln_1()
# %%
def grad_exp_1():
    import torch
    a = torch.from_numpy(np.arange(1, 7, dtype=np.float32).reshape((6, )))
    a.requires_grad = True
    b = a.exp()
    loss = b.mean()
    print(b, b.shape, loss, sep="\n")
    loss.backward()
    print_arr(a)
    print_arr(b, dig=3)
    print_arr(a.grad, dig=3)
grad_exp_1()
# %%
def grad_exp_2():
    import torch
    a = torch.from_numpy(np.arange(1, 7, dtype=np.float32).reshape((6, )))
    a.requires_grad = True
    b = a.exp()
    loss = b.sum()
    print(b, b.shape, loss, sep="\n")
    loss.backward()
    print_arr(a)
    print_arr(b, dig=3)
    print_arr(a.grad, dig=3)
grad_exp_2()
# %%
def grad_logsofmax_2():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = d.sum(dim=1, keepdim=True)
    f = e.log()
    g = c - f
    gg = a.log_softmax(1)
    loss = g.sum()
    loss.backward()
    print(a, b, c, d, e, f, g, gg, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    print_arr(e, dig=4)
    print_arr(f, dig=4)
    print_arr(g, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_logsofmax_2()
# %%
def grad_logsofmax_3():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = d.sum(dim=1, keepdim=True)
    f = e.log()
    ff = f.broadcast_to(((2, 3)))
    loss = ff.sum()
    loss.backward()
    print(a, b, c, d, e, f, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    print_arr(e, dig=4)
    print_arr(f, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_logsofmax_3()

# %%
def grad_logsofmax_4():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = d.sum(dim=1, keepdim=True)
    f = e.log()
    ff = f.broadcast_to(((2, 3)))
    g = c.sub(ff)
    gg = a.log_softmax(1)
    loss = g.sum()
    loss.backward()
    print(a, b, c, d, e, f, ff, g, gg, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    print_arr(e, dig=4)
    print_arr(f, dig=4)
    print_arr(g, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_logsofmax_4()
# %%
def grad_sub_1():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    loss = c.sum()
    loss.backward()
    print(a, b, c, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_1()
# %%
def grad_sub_2():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = c.sub(d)
    loss = e.sum()
    loss.backward()
    print(a, b, c, d, e, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_2()
# %%
def grad_broadcast_2():
    import torch
    a = torch.from_numpy(np.arange(0, 3, dtype=np.float32).reshape((3, 1)))
    a.requires_grad = True
    b = a.broadcast_to(((3, 2)))
    loss = b.sum()
    loss.backward()
    print(a, b, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)
grad_logsofmax_2()
# %%
# %%
def grad_sub_3():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    loss = c.sum()
    loss.backward()
    print(a, b, c, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_3()

# %%
def grad_sub_4():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.exp()
    c = a.sub(b)
    loss = c.sum()
    loss.backward()
    print(a, b, c, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_4()


# %%
def grad_sub_5():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = d.sum(dim=1, keepdim=True)
    f = d.div(e)
    # ff = a.softmax(1)
    loss = f.sum()
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, c, d, e, f, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    print_arr(e, dig=4)
    print_arr(f, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_5()

# %%
def grad_sub_6():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = d.sum(dim=1, keepdim=True)
    loss = e.sum()
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, c, d, e, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    print_arr(e, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_6()


# %%
def grad_sub_7():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.sub(b)
    d = c.exp()
    e = c.div(d)
    # ff = a.softmax(1)
    loss = e.sum()
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, c, d, e, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    print_arr(e, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_7()



# %%
def grad_sub_8():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.max(dim=1, keepdim=True).values
    c = a.div(b)
    loss = c.sum()
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, c, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_8()


# %%
def grad_sub_9():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.exp()
    c = a.div(b)
    loss = c.sum()
    c.register_hook(lambda d: print_arr(d))
    b.register_hook(lambda d: print_arr(d, dig=4))
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, c, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

grad_sub_9()



# %%
def test_tensor_grad_sqrt():
    import torch
    a = torch.from_numpy(np.arange(1, 7, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.sqrt()
    loss = b.sum()
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

test_tensor_grad_sqrt()

# %%
def test_narrow_1():
    import torch
    a = torch.from_numpy(np.arange(1, 61, dtype=np.float32).reshape((3, 5, 4)))
    b = a.narrow(0, 1, 2)
    c = a.narrow(1, 2, 3)
    d = a.narrow(2, 0, 3)
    print_arr(a)
    print_arr(b)
    print_arr(c)
    print_arr(d)
test_narrow_1()
# %%
def test_grad_concat_1():
    import torch
    a = torch.from_numpy(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
    a.requires_grad = True
    b = a.exp()
    c = a.div(b)
    d = torch.concat([a, c], dim=1)
    loss = d.sum()
    d.register_hook(lambda d: print_arr(d))
    c.register_hook(lambda d: print_arr(d))
    b.register_hook(lambda d: print_arr(d, dig=4))
    loss.backward()
    # loss = ff.sum()
    # loss.backward()
    print(a, b, c, loss, sep="\n")
    print_arr(a)
    print_arr(b, dig=4)
    print_arr(c, dig=4)
    print_arr(d, dig=4)
    # print_arr(ff, dig=4)
    print(loss)
    print_arr(a.grad, dig=4)

test_grad_concat_1()


# %%
