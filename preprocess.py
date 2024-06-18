import ast
import os
from ast2vec import ast2vec
from ast2vec import python_ast_utils
from scipy import fft 
import torch
import numpy as np

path = "Method3/data"
path2 = "Method3/test"
trees = []  # A list where we are going to store the AST's
X = []  # A list with all the vectors
Y = [] # A list with all the vectors after DCT
testTrees = [] # A list where we are going to store the test AST's
testVec = [] # A list with all the test vectors
testDCT = [] # A list with all the test vectors after DCT

# Reading all the codes as .txt and parsing them
for programs in os.listdir(path):
    with open(path + "/" + programs, "r") as file:
        program = file.read()
    trees.append(python_ast_utils.ast_to_tree(ast.parse(program)))

# Loading the model
model = ast2vec.load_model()

# Generating the code vectors and storing in X
for tree in trees:
    X.append(model.encode(tree))

# By now, we have in X the vectors after ast2vec
# We want to apply DCT to all vectors in X
def compute_dct(vector):
    return fft.dct(vector, norm='ortho')

# As the vectors in X are tensors and have gradients attached, we need to
# detach and transform in a numpy array
for vec in X:
    Y.append(compute_dct(vec.detach().numpy()))

# Reading all the test codes as .txt and parsing them
for programs in os.listdir(path2):
    with open(path2 + "/" + programs, "r") as file:
        program = file.read()
    testTrees.append(python_ast_utils.ast_to_tree(ast.parse(program)))

# Generating the code vectors and storing in testVec
for tree in testTrees:
    testVec.append(model.encode(tree))

for vec in testVec:
    testDCT.append(compute_dct(vec.detach().numpy()))

# Checking the distance betwen two dct vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Calculate cosine similarity between vector_to_compare and each initial vector
similarities1 = [cosine_similarity(testDCT[0], dct_vec) for dct_vec in Y]
similarities2 = [cosine_similarity(testDCT[1], dct_vec) for dct_vec in Y]

# Find the index of the vector with the highest similarity
closest_index1 = np.argmax(similarities1)
closest_vector1 = Y[closest_index1]

closest_index2 = np.argmax(similarities2)
closest_vector2 = Y[closest_index2]

print(f"The first test vector is closest to the {closest_index1+ 1}th initial vector.")
print(f"Cosine similarity: {similarities1[closest_index1]}")

print(f"The second test vector is closest to the {closest_index2+ 1}th initial vector.")
print(f"Cosine similarity: {similarities2[closest_index2]}")

# print(X[4])
# print(Y[4])
# print(trees[0])
