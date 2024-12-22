# 卒業研究
## テーマ：量子ネットワークでの正確な情報の送受信について
研究内容

量子情報を送受信する際には、ノイズや時間経過などにより、その量子が持つ情報が失われて行ってしまう。
そのため、できるだけ送信する前の元の状態の情報をどれだけ正確に伝送することができるかということを調べようとした。

どれだけ正確に伝送されているかを示す指標がフィデリティというもので、これは、量子情報の類似度を表すものである。
フィデリティは0～1の値を取り、1に近いほど、比較している量子は近い状態にあるということだ。
つまり、送信前の量子と受信後の量子のフィデリティを計測することで、どれだけ近い状態にあるか、すなわち正確に伝送できているかを知ることができる。

このフィデリティを向上させるための方法がいくつかあるが、本研究では、フィルタリングと蒸留という方法を用いることにした。

## 進捗
### ~12/20

＠sim2.py

・フィルタリング操作の後に、テレポーテーションを加えようとしている。

　->とりあえず正しい順序で操作が行われていることを確認

### 12/21
 
@sim2.py

・各ノードで、テレポーテーションするための適切な操作を加えるコードの記述

### 12/22

@sim2.py

・エラーの修正

・エンタングルメント生成 -> フィルタリング -> テレポーテーション　の大枠は完成
　
　（1回の試行が終了した時に、量子メモリを開放する部分の調整が必要）
