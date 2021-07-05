

def modelMutip(m, c=1):
    for j in m.state_dict():
        m.state_dict()[j].copy_(m.state_dict()[j] * c)
    return m

def addTwoModel(m1, m2, weight1=1, weight2=1):
    # m3=copy.deepcopy(m1)
    for j in m2.state_dict():
        # for j,_ in m2.named_parameters():
        m1.state_dict()[j].copy_(m1.state_dict()[j] * weight1 + m2.state_dict()[j] * weight2)
    return m1

def wswaUpdate(mn1, Sum_an_n, an1, wn, w=0):
    '''
    :param mn1: m_(n+1)
    :param Sum_an_n: sum i=1:n (a_i - w)
    :param an1:a_(n+1)
    :param wn:previous wswa model
    :param w:hyperparameter
    :return:wn1 next wswa model
    Sum_an_n1  sum i=1:n+1 (a_i - w)

    '''
    up = addTwoModel(wn, mn1, Sum_an_n, an1 - w)
    down = Sum_an_n + an1 - w
    wn1 = modelMutip(up, 1 / down)
    Sum_an_n1 = Sum_an_n + an1 - w
    return wn1, Sum_an_n1

