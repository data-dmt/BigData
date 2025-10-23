from src.matrix import matmul

def test_small_case():
    A = [[1,2],[3,4]]
    B = [[5,6],[7,8]]
    C = matmul(A,B,"ikj")
    assert C == [[19.0,22.0],[43.0,50.0]]
