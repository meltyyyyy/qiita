# Qiita

Qiitaへ投稿した記事の中で用いたプログラムを置いているレポジトリです。

## [【MCMC】メトロポリス・ヘイスティングス法、ハミルトニアンモンテカルロ法、ギブスサンプリングを比較する](https://qiita.com/meltyyyyy/items/b04e5c13a0ea71c2be05)

**マルコフ連鎖モンテカルロ法(Markov chain Monte Carlo methods)** は、解析的に計算することが難しい事後分布からサンプリングしたいときに用いられる手法です。この記事ではMCMC法を用いたサンプリングアルゴリズム(M-H法、HMC法、ギブスサンプリング)を紹介しました。2次元ガウス分布を目標分布として、3つのアルゴリズムの振る舞いや性能を比較を行いました。

<img src=https://user-images.githubusercontent.com/81362789/180633664-0b70caf5-cc49-4801-add1-552348506dec.png width=400>

## [ベイズ最適化 from Scratch](https://qiita.com/meltyyyyy/items/e67f22f98a96e30e5461)

非自明な関数の最大値、または最小値を求める手法は **Black-Box最適化** と呼ばれます。Black-Box最適化の中でもベイズの枠組みを用いて、関数の最大値、または最小値を求めていく方法を、 **ベイズ最適化(Bayesian Optimization)** と呼びます。この記事ではベイズ最適化を実行するプログラムをゼロから実装し、ベイズ最適化のアルゴリズムへの理解を深めました。

![bo](https://user-images.githubusercontent.com/81362789/180634525-8dbcd42b-2896-44fe-85d1-451ab4984205.gif)

## [ベイズ最適化 from Scratch](https://qiita.com/meltyyyyy/items/e67f22f98a96e30e5461)

非自明な関数の最大値、または最小値を求める手法は **Black-Box最適化** と呼ばれます。Black-Box最適化の中でもベイズの枠組みを用いて、関数の最大値、または最小値を求めていく方法を、 **ベイズ最適化(Bayesian Optimization)** と呼びます。この記事ではベイズ最適化を実行するプログラムをゼロから実装し、ベイズ最適化のアルゴリズムへの理解を深めました。

![bo](https://user-images.githubusercontent.com/81362789/180634525-8dbcd42b-2896-44fe-85d1-451ab4984205.gif)

## ガウス過程

### [ガウス過程 from Scratch](https://qiita.com/meltyyyyy/items/8440849532cd55da1e45)

### [ガウス過程 from Scratch MCMCと勾配法によるハイパーパラメータ最適化](https://qiita.com/meltyyyyy/items/5a058ecc81e010876a39)

ガウス過程を用いる場合とき、通常は事前に最適なハイパーパラメータは分かりません。

この記事では、ガウス過程のハイパーパラメータ最適化に関する数式を導出し、最適化を実行するプログラムをゼロから実装を行いました。

一般的にガウス過程のハイパーパラメータ最適化の文脈で用いられるのは、　**マルコフ連鎖モンテカルロ法(Markov chain Monte Carlo method)** か **勾配法(Gradient Decent method)** です。この記事では、MCMCによる最適化と勾配法による最適化の両方の実装を行いました。

### [ガウス過程 from Scratch コレスキー分解による高速化](https://qiita.com/meltyyyyy/items/44e2f270be72943086f3)

特に何も工夫をしないままのガウス過程ではトータルの計算量として $O(N3)$ を必要としてしまいます。このままでは $N$ が小さいうちは大きな問題にはなりませんが、 $N$ が大きくなると手に途端に負えなくなってしまいます。

この記事では **コレスキー分解(Cholesky decomposition)** を用いることでガウス過程の計算量を $O(N3)$ から $O(N2)$ まで削減していきます。

### [ガウス過程 from Scratch Non-Gaussianな尤度によるガウス過程](https://qiita.com/meltyyyyy/items/620691c0cd07023777cc)

通常のガウス過程では、関数 $\mathbf{f}$ と出力 $\mathbf{y}$　の関係 $P(\mathbf{y}|\mathbf{f})$ がガウス分布 $\mathbb{N}(\mathbf{f},\sigma^2\mathbf{I})$ に従うという前提のもと、出力を計算していました。

この記事では、尤度 $P(\mathbf{y}|\mathbf{f})$ がコーシー分布に従う場合を考え、データに予期しない外れ値が含まれていてもうまく回帰できるようなロバストなガウス過程を実装しました。

### [ガウス過程力学モデルと主成分分析、ガウス過程潜在変数モデルを比較する](https://qiita.com/meltyyyyy/items/f2e9f81354d1ed72a5d1)

**ガウス過程力学モデル(Gaussian Process Dynamical Model)** はガウス過程による教師なし学習の一つで、 **ガウス過程潜在モデル(Gaussian Process Latent Variable Model)** を拡張したモデルのうちの一つです。

GPLVMでは、潜在変数 $\mathbf{X}=(\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_N)$ の独立性を仮定していました。これに対してGPDMは、潜在変数 $\mathbf{X}$ が時系列データであるという仮定を導入して、潜在空間での構造を学習します。

GPDMの原論文を参考にしながら、GPDMをゼロから実装し、PCAやGPLVMなどの他の次元圧縮手法と比較を行いました。
