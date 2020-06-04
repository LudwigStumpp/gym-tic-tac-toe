
def base_x_to_dec(base_list, x):
    return int(''.join(map(str, base_list)), base=x)


def dec_to_base_x(dec, x):
    base_list = []

    left = dec
    while left > 0:
        base_list.append(left % x)
        left = left // x

    return list(reversed(base_list))


def list_to_matrix(list, n):
    return [list[r::n] for r in range(n)]
