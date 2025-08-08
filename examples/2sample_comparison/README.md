# Comparing EagleEye and various two-sample tests
In `2samp_tests.ipynb`, we provide a few examples examples to demosntrate where one would like to use EE instead of an MLP,BDT,wavelet, or vanilla kNN classifier. 

-------

## How are MLP classifiers supposed to work in this case?   


Optimising binary cross-entropy minimises: 
$$\mathcal{L}(f)=-\mathbb{E}_{x \sim p_{\mathrm{ub}}}[\log (1-f(x))]-\mathbb{E}_{x \sim p_{\text {date }}}[\log f(x)] .$$

Assuming the universal approximation holds for the NN $f$ optimal solution is

$$
f(x)=\operatorname{Pr}(Y=1 \mid x)=\frac{\pi_1 p_{\text {data }}(x)}{\pi_0 p_{\text {bg }}(x)+\pi_1 p_{\text {data }}(x)} \tag{A}
$$

Assuming equal priors

$$
\Rightarrow \mathcal{L}(x) \sim \frac{p_{\text {data }}(x)}{p_{\text {bg }}(x)}
$$

If then the data is a mixture of signal + bkg, the real data distribution is:

$$
p_{\text {data }}(x)=(1-\varepsilon) p_{\text {bg }}(x)+\varepsilon p_{\text {sig }}(x)\tag{B} .
$$

Then the true density ratio is

$$
r(x)=\frac{p_{\text {data }}(x)}{p_{\text {bg }}(x)}=1-\varepsilon+\varepsilon \frac{p_{\text {sig }}(x)}{p_{\text {bg }}(x)} .
$$

and is used as the anomaly detection threshold (see CATHODE paper).

-------

In this example case, we have $\epsilon<<1$ and the following distributions:

* **Background** (label \(y=0\))  
  $$p_{\mathrm{bg}}(x)=\mathcal N\!\bigl(0,I_d\bigr)$$  

* **Signal** (label \(y=1\))  
  $$p_{\mathrm{sig}}(x)=\mathcal N\!\bigl(0,\sigma^{2}I_d\bigr), \qquad \sigma= 0.05$$  



Recall (subbing B into A)

$$
\Pr(Y=1\mid x)
       =\frac{\varepsilon\,p_{\mathrm{sig}}(x)}
              {(1-\varepsilon)\,p_{\mathrm{bg}}(x)
               +\varepsilon\,p_{\mathrm{sig}}(x)}.
\tag{1}
$$




For the centred Gaussians the only difference at \(x=0\) is the normalisation constant becasue the bkg follows a std normal:

$$
p_{\mathrm{bg}}(0)=\frac1{(2\pi)^{d/2}},
\qquad
p_{\mathrm{sig}}(0)=\frac1{(2\pi\sigma^{2})^{d/2}}
                   =\sigma^{-d}\,p_{\mathrm{bg}}(0),
$$

where $d$ is the number of dimentions (here $d=20$).

so  
$$
\frac{p_{\mathrm{sig}}(0)}{p_{\mathrm{bg}}(0)}=\sigma^{-d}.
\tag{2}
$$




Insert (2) into (1) to obtain the true estimate for the posterior at the origin:

$$
f^T(0)\sim\Pr(Y=1\mid x=0)
   =\frac{\varepsilon\,\sigma^{-d}}
          {(1-\varepsilon)+\varepsilon\,\sigma^{-d}}
   =\boxed{\displaystyle
     \frac{\varepsilon/\sigma^{d}}
          {1-\varepsilon+\varepsilon/\sigma^{d}} }.
\tag{3}
$$

This $\rightarrow 1$ as $d\rightarrow$ large, as expected since we have a very concentrated anomaly!


## Vanishing-gradient effect  
The above will break down when an anomolous overdensity is represented by a very small number of points, swamped by a very large backgorund. 

With a batch of size $B$ the gradient step on a parameter vector $w$ is

$$
\Delta w=-\eta \frac{1}{B} \sum_{j=1}^B\left[f\left(x_j ; w\right)-y_j\right] \frac{\partial z_j}{\partial w}, \tag{4}
$$

where $f = \text{sigmoid}(x)$ is the NN prediction and $\eta$ is the learning rate. For batch sizes of size, say, $B=128$, we will very rarely draw a point with label $y=1$ since, for a 20d Gaussian, the probability of getting a a point within $\sigma = 0.05$ is

$p = \chi^2_\text{CDF} \sim 10^{-36}$.

As  result, the sum in Eqn 4 will not have many contributions from points with label $y=1$. Therefore, for weights whose $\partial z / \partial w$ is non-zero only inside this region, the per-batch gradient is very close 0 unless the batch happens to include at least one sample from $R$.

But  even when one such sample sneaks in, $|f-y| \leq 1$, so the contribution to the update is bounded by $\eta / B$.

Hence the expectation over batches

$$
\mathbb{E}[\|\Delta w\|] \lesssim\eta \sigma^d
$$

which for $d=20$  is astronomically smaller than the typical $O(\eta)$ updates for weights that are driven by the bulk of the data.

## 20-d Gaussian with concentrated density anomaly at the origin - "Needle in a haystack" a.k.a "most anomaly detection tasks in physics?"

Here we just consider a 20D gaussian with another very tight gaussian at the origin representing the signal ($n_\text{sig}\sim100$ points). 

The reference set contain 100k of the bkg points and teh test has the other 100k plus the 100 signals.

```python
d      = 20
n_bg   = 200_000
n_sig  = 100         

X_bg   = np.random.randn(n_bg, d)
X_sig  = np.random.randn(n_sig, d) * sigma   # much tighter

# labels: 0 = background, 1 = signal 
X_data = np.vstack([X_bg[:len(X_bg)//2], X_sig])
labels = np.hstack([np.zeros(len(X_bg)//2), np.ones(n_sig)])
X_ref  = X_bg[len(X_bg)//2:]
```
![Needle in a haystack: 2D projection](./MLP_comp_plots/haystack_data.png)



## EagleEyE Results 
TIME TAKEN TO RUN Macbook M1 (Putative + IDE pruning): **1m 1.2s**

-----------
EagleEye sees $\sim100$% of anomolous points within the pruned set ($p_\text{ext}= 10^{-5} \rightarrow \Upsilon^*_+ \sim 38$)

![Needle in a haystack: 2D projection](./MLP_comp_plots/eagleeye_scatter_01_23.png)
![Needle in a haystack: 2D projection](./MLP_comp_plots/eagleeye_scatter_45_67.png)

![Needle in a haystack: 2D projection](./MLP_comp_plots/eagleeye_null_vs_test_hist.png)

Clearly an anomaly...needs pruning tho!

Also inspect the faction of all points above $\Upsilon_+$ as a function of  $\Upsilon_+$ for both putative and pruned sets: Pruning recovers basically all points contributing to anomolous density. 


<p align="center">
  <img src="./MLP_comp_plots/eagleeye_fraction_vs_threshold.png" alt="Fraction vs threshold" width="48%" style="display:inline-block;"/>
  <img src="./MLP_comp_plots/eagleeye_fraction_vs_threshold_pruned.png" alt="Fraction vs threshold (pruned)" width="48%" style="display:inline-block;"/>
</p>


ROC curve: AUC = 0.9999719 (Using true labels)

![Needle in a haystack: 2D projection](./MLP_comp_plots/eagleeye_roc_curve.png)

![Needle in a haystack: 2D projection](./MLP_comp_plots/eagleeye_recall_precision_accuracy_vs_threshold.png)

![Needle in a haystack: 2D projection](./MLP_comp_plots/eagleeye_recall_precision_accuracy_vs_threshold_pruned.png)



## MLP Results 

##### MLP trained to classify bkg vs data (not signal) estimates the likelihood ratio: 

------------

TIME TAKEN TO RUN macbook M1 GPU: **5m 16.2s**

The neural network of course requires extensive hyperparameter tuning, as opposed to a kNN based method like EagleEye that is deterministic and works out of the box. 

- **Input:**  
  - Concatenated reference and data samples, each with `d=20` features.

- **Network Architecture:**  
  - **Input layer:** Shape = number of features (`d=20`)
  - **Hidden layers:** 3 fully connected layers with 64 units each, ReLU activation:
    - Dense(64, activation="relu")
    - Dense(64, activation="relu")
    - Dense(64, activation="relu")
  - **Output layer:**  
    - Dense(1, activation="sigmoid")  

- **Training:**  
  - **Loss:** Binary cross-entropy
  - **Optimizer:** Adam (learning rate = 1e-3)

  - **Validation split:** 20% of training data
  - **Epochs:** 30 (with early stopping)
  - **Batch size:** 128


-----------
Absolutely shits the bed

![Needle in a haystack: 2D projection](./MLP_comp_plots/mlp_scatter_01_23.png)
![Needle in a haystack: 2D projection](./MLP_comp_plots/mlp_scatter_45_67.png)

Distribution of the likelihood ratio estimate

![Needle in a haystack: 2D projection](./MLP_comp_plots/mlp_lr_hist.png)



![Needle in a haystack: 2D projection](./MLP_comp_plots/mlp_fraction_vs_threshold.png)

...and the roc curve - not sure whats happening here. Still debugging

ROC curve: AUC = 0.7782 (Using true labels)

![Needle in a haystack: 2D projection](./MLP_comp_plots/mlp_roc_curve.png)

![Needle in a haystack: 2D projection](./MLP_comp_plots/mlp_recall_precision_accuracy_vs_threshold.png)




