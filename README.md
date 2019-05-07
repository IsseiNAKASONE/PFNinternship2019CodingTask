# PFN Internship 2019 Coding Task
## 動作環境
-使用言語：python3.7.3<br>
-OS：macOS Mojave10.14.4
## 概要
本ソースコードはGraph Neural Networkをchainerライブラリをインスパイアして実装したものである．<br>
課題1〜4はそれぞれmain1.py〜main4.pyに対応しており，コマンドライン上で<br>
```
$ python3 main1.py
```
などと実行すればよい．
コマンドライン引数は一切なく，ベクトルの初期化，入力データセットへのファイルパスなどは
各ソースコード内に直に記述している．<br>
ディレクトリ配置は以下を想定している．<br><br>
┣ datasets/<br>
┃┣ train/<br>
┃┗test/<br>
┗ src/<br>
&emsp;┣ prediction.txt<br>
&emsp;┗ *.py<br><br>
`main3.py`，`main4.py`を実行すると各エポックごとの損失値とaccuracyのログファイルが
カレントディレクトリに`log.json`の名前で出力されるようになっている．
この出力ファイル名は関数引数で指定可能である．<br>
例えば，同じ条件で何回か学習を実行し，そのログファイル`log_1.json`,`log_2.json`,...が`./foo/`ディレクトリ下
にあるとする．このとき，<br>
```
$ python3 plot2d.py -i foo
```
を実行すると，平均損失，平均accuracyのグラフがエラーバー付きで`foo`下に出力される．
また，`main4.py`についてはテストデータに対する予測ラベルが`prediction.txt`として出力される．
 