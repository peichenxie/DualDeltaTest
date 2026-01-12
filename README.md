# Dual-Delta Test for Quantifying Numerical Accuracy

Suppose you implemented a custom mixed-precision function impl_1:
- maybe a fused operator,
- a hand-optimized GPU kernel,
- or a quantized inference path.

You want to answer:
- Is my implementation numerically correct?
- How does it compare to a baseline reference impl_2?

The Dual-Delta Test is a systematic methodology for evaluating the numerical accuracy of your custom mixed-precision function. Instead of relying on a single error metric against the reference implementation (single delta between impl_1 and impl_2), the Dual-Delta Test measures two error distributions (dual deltas) against the oracle.

```python
def dual_delta_test(impl_1, impl_2, oracle, generate_input, get_error, num_tests):
    delta_1 = []  # Error of your custom implementation against the oracle
    delta_2 = []  # Error of the reference mixed-precision implementation against the oracle
    for _ in range(num_tests):
        input = generate_input()
        res_1 = impl_1(*input)
        res_2 = impl_2(*input)
        res_oracle = oracle(*input)
        delta_1.append(get_error(res_1, res_oracle))
        delta_2.append(get_error(res_2, res_oracle))
    return delta_1, delta_2
```

By comparing these distributions, you can determine whether your custom function performs as well as, better than, or worse than the baseline reference.

## Example

```python
# Suppose you have a custom half-precision matrix multiplication function on GPU
def matmul(A, B):
    # using pytorch's GPU matmul as an example
    return torch.matmul(A.cuda(), B.cuda()).cpu()

# You can use pytorch's CPU matmul as the reference half-precision implementation
def matmul_ref(A, B):
    return torch.matmul(A, B)

# You can define an oracle using high-precision floating-point operations
def matmul_oracle(A, B):
    return torch.matmul(A.double(), B.double())
```

You can then set up the Dual-Delta Test as follows:

```python
def generate_input():
    A = torch.randn(128, 128).half()
    B = torch.randn(128, 128).half()
    return A, B

# You can choose any error metric, such as max absolute/relative error, mean squared error, ULP error, etc.
def get_error(res, res_oracle):
    # using Max Hyb Error (https://github.com/peichenxie/HybError) as an example
    A = res.double()
    B = res_oracle.double()
    diff = (A-B).abs() / (1+B.abs())
    return diff.max()

delta_1, delta_2 = dual_delta_test(impl_1=matmul, impl_2=matmul_ref, oracle=matmul_oracle, generate_input=generate_input, get_error=get_error, num_tests=1000)
```

Finally, you can analyze the results through means, variances, histograms, or statistical tests to draw conclusions about the numerical accuracy of your custom implementation compared to the reference.

```python
# Example analysis: visualizing the error distributions through histograms
import matplotlib.pyplot as plt

plt.hist(delta_1, bins=50, alpha=0.5, label='Custom Implementation')
plt.hist(delta_2, bins=50, alpha=0.5, label='Reference Implementation')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Error Distribution Comparison')
plt.show()
```
