# Qiita

Qiitaへ投稿した記事の中で主に機械学習系の再現コードを置いているレポジトリです。


## [【MCMC】メトロポリス・ヘイスティングス法、ハミルトニアンモンテカルロ法、ギブスサンプリングを比較する](https://qiita.com/meltyyyyy/items/b04e5c13a0ea71c2be05)

**マルコフ連鎖モンテカルロ法(Markov chain Monte Carlo methods)** は、解析的に計算することが難しい事後分布からサンプリングしたいときに用いられる手法です。この記事ではMCMC法を用いたサンプリングアルゴリズム(M-H法、HMC法、ギブスサンプリング)を紹介しました。2次元ガウス分布を目標分布として、3つのアルゴリズムの振る舞いや性能を比較を行いました。

<img src=https://user-images.githubusercontent.com/81362789/180633664-0b70caf5-cc49-4801-add1-552348506dec.png>

## [ベイズ最適化 from Scratch](https://qiita.com/meltyyyyy/items/e67f22f98a96e30e5461)

非自明な関数の最大値、または最小値を求める手法は **Black-Box最適化** と呼ばれます。Black-Box最適化の中でもベイズの枠組みを用いて、関数の最大値、または最小値を求めていく方法を、 **ベイズ最適化(Bayesian Optimization)** と呼びます。この記事ではベイズ最適化を実行するプログラムをゼロから実装し、ベイズ最適化のアルゴリズムへの理解を深めました。

![bo](https://user-images.githubusercontent.com/81362789/180634525-8dbcd42b-2896-44fe-85d1-451ab4984205.gif)

## ガウス過程シリーズ

### [ガウス過程 from Scratch](https://qiita.com/meltyyyyy/items/8440849532cd55da1e45)

**ガウス過程(Gaussian Process)** はノンパラメトリックなモデルの一つで、一般的に「関数を出力する箱」というような例えられ方をします。このようにたとえられる理由は、ガウス過程では関数 **f(x)** を確率変数と見立てて、 **f(x)** の確率分布を出力するためです。

ガウス過程は、ベイズ最適化や空間統計学の文脈でよく用いられます。また、最近になって深層学習が発展するにつれて、ニューラルネットワークとガウス過程の等価性が示され(Lee et al, 2018)、注目を集めるようにもなりました。具体的には、隠れ層が１層のニューラルネットワークで隠れ層のユニット数を∞にすると、中心極限定理によりニューラルネットワークの出力はガウス過程と等価になります。

この記事では、ガウス過程の導出とプログラムのスクラッチ実装を行いました。

![image](https://user-images.githubusercontent.com/81362789/187078942-8b84644e-7293-4ed6-aa83-9d7e17b4c5d8.png)

### [ガウス過程 from Scratch MCMCと勾配法によるハイパーパラメータ最適化](https://qiita.com/meltyyyyy/items/5a058ecc81e010876a39)

ガウス過程を用いる場合とき、通常は事前に最適なハイパーパラメータは分かりません。

この記事では、ガウス過程のハイパーパラメータ最適化に関する数式を導出し、最適化を実行するプログラムをゼロから実装を行いました。

一般的にガウス過程のハイパーパラメータ最適化の文脈で用いられるのは、　**マルコフ連鎖モンテカルロ法(Markov chain Monte Carlo method)** か **勾配法(Gradient Decent method)** です。この記事では、MCMCによる最適化と勾配法による最適化の両方の実装を行いました。

### [ガウス過程 from Scratch コレスキー分解による高速化](https://qiita.com/meltyyyyy/items/44e2f270be72943086f3)

特に何も工夫をしないままのガウス過程ではトータルの計算量として <img src="https://latex.codecogs.com/gif.latex?O(N^3)" /> を必要としてしまいます。このままではNが小さいうちは大きな問題にはなりませんが、Nが大きくなると手に途端に負えなくなってしまいます。

この記事では **コレスキー分解(Cholesky decomposition)** を用いることでガウス過程の計算量を <img src="https://latex.codecogs.com/gif.latex?O(N^3)" /> から <img src="https://latex.codecogs.com/gif.latex?O(N^2)" /> まで削減していきます。

### [ガウス過程 from Scratch Non-Gaussianな尤度によるガウス過程](https://qiita.com/meltyyyyy/items/620691c0cd07023777cc)

通常のガウス過程では、関数 <img src="https://latex.codecogs.com/gif.latex?\mathbf{f}" /> と出力 <img src="https://latex.codecogs.com/gif.latex?\mathbf{y}" /> の関係 <img src="https://latex.codecogs.com/gif.latex?P(\mathbf{y}|\mathbf{f})" /> がガウス分布 <img src="https://latex.codecogs.com/gif.latex?\mathbb{N}(\mathbf{f},\sigma^2\mathbf{I})" />　に従うという前提のもと、出力を計算していました。

この記事では、尤度 <img src="https://latex.codecogs.com/gif.latex?P(\mathbf{y}|\mathbf{f})" /> がコーシー分布に従う場合を考え、データに予期しない外れ値が含まれていてもうまく回帰できるようなロバストなガウス過程を実装しました。

![image](https://user-images.githubusercontent.com/81362789/187079456-da19fb97-69d9-47c4-88fe-562fa995a1f0.png)

### [ガウス過程力学モデルと主成分分析、ガウス過程潜在変数モデルを比較する](https://qiita.com/meltyyyyy/items/f2e9f81354d1ed72a5d1)

**ガウス過程力学モデル(Gaussian Process Dynamical Model)** はガウス過程による教師なし学習の一つで、 **ガウス過程潜在モデル(Gaussian Process Latent Variable Model)** を拡張したモデルのうちの一つです。

GPLVMでは、潜在変数 <img src="https://latex.codecogs.com/gif.latex?\mathbf{X}=(\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_N)" /> の独立性を仮定していました。これに対してGPDMは、潜在変数 <img src="https://latex.codecogs.com/gif.latex?\mathbf{X}" /> が時系列データであるという仮定を導入して、潜在空間での構造を学習します。

この記事では、GPDMの原論文を参考にしながら、GPDMをゼロから実装し、PCAやGPLVMなどの他の次元圧縮手法と比較を行いました。

![image](https://user-images.githubusercontent.com/81362789/187079576-bb594b2b-81cd-4520-9be9-246965a733c5.png)

## [シミュレーテッド・アニーリング from Scratch](https://qiita.com/meltyyyyy/items/096efb08fb4ec532c330)

**シミュレーテッド・アニーリング(Simulated Annealing)** は、「焼きなまし法」とも呼ばれ、大域的最適化問題へのアプローチ方法の一つです。「焼きなまし」の名称から察せられる通り、金属を高温の状態にして、徐々に温度を下げることで秩序がある構造を作り出す焼きなましの技術をコンピュータ上で再現したアルゴリズムになります。

この記事では、シミュレーテッド・アニーリングのスクラッチ実装を行いました。

![image](https://user-images.githubusercontent.com/81362789/187080038-86ff2aaf-4df0-4549-be4e-d8cd3447b12c.png)

## [馬蹄分布(Horseshoe Distribution)の基本](https://qiita.com/meltyyyyy/items/91a95a777c6c6e6c7e5a)

馬蹄分布は確率変数がスパースであるという事前知識があるときに用いられる分布になっています。

馬蹄分布そのものは平均0のある分散の値を持った正規分布から生成されるものですが、縮小係数は τ=1 のときにBeta分布から生成され、その形が馬の蹄に似ていることから馬蹄分布という名前がつきました。

この記事では馬蹄分布の基本的な事柄について解説を行いました。

![horseshoe](https://user-images.githubusercontent.com/81362789/190842274-ebffbcd9-114b-4152-8efa-141ecfc04aa1.png)

