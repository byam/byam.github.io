---
layout: post
title:  "Data Engineering Glossary"
date:   2018-06-11 00:00:00 +0900
categories: glossary, data
fbcomments: true
---

Glossary:

- [Serialize and Deserialize](#serialize-and-deserialize)
- [Partitioning, Sharding, Replication](#partitioning-sharding-replication)
- [ETL](#etl)
- [Batch and Real-Time processing](#batch-and-real-time-processing)
- [Database transaction](#database-transaction)
- [ACID](#acid)


## 

![]({{ "/assets/img/de-glossary/" | absolute_url }}){: .center-image }{:width="600px"}

### References
- []()




## Serialize and Deserialize

### Serialize
データ(オブジェクト)を特定の形式(バイト列)に変換し、ファイル(データベース)に格納する、または変換されたデータをネットワークに転送するプロセス。

### Deserialize
Serializeの逆で、変換されたデータをデータに復元するプロセス。


![serialize-deserialize]({{ "/assets/img/de-glossary/serialize-deserialize.jpg" | absolute_url }}){: .center-image }{:width="600px"}

### References
- [Serialization](https://en.wikipedia.org/wiki/Serialization)
- [Javaのシリアライズ(直列化)メモ](https://qiita.com/Sekky0905/items/b3c6776d10f183d8fc89)




## Partitioning, Sharding, Replication

### Partitioning
データ要素を複数のエンティティに分割すること。

### Sharding(horizontal partitioning)
スキーマが共通で、データの範囲を決めて、データを分割すること。

### Vertical partitioning
スキーマを切り分けて、あるエンティティに格納されたデータを複数のエンティティに分割すること。

### Replication
複数のノードでデータを複製する方法


![replica-partition.jpg]({{ "/assets/img/de-glossary/replica-partition.jpg" | absolute_url }}){: .center-image }{:width="600px"}

### References
- [What's the difference between sharding DB tables and partitioning them?](https://www.quora.com/Whats-the-difference-between-sharding-DB-tables-and-partitioning-them)



## ETL

**ETL**(extract, transform and load) とは、データウェアハウスにデータを入れること。

![etl.png]({{ "/assets/img/de-glossary/etl.png" | absolute_url }}){: .center-image }{:width="600px"}

### References
- [What is　ETL](https://www.quora.com/What-is-ETL)



## Batch and Real-Time processing

### Batch Process
すべての入力を受け取り、指定された時間および出力を完了した後にデータ処理すること。

### Real-Time process
入力が受信されるとすぐにデータ処理すること。



## Database transaction

**トランザクション**は、最小処理単位とみなされるタスクの集合である。

**データベーストランザクション**とは、トランザクションアクションに対して、全てのタスクを処理されるか、まったく処理されないかのどちらかです。

![database-transaction.png]({{ "/assets/img/de-glossary/database-transaction.png" | absolute_url }}){: .center-image }{:width="600px"}


## ACID

**ACID**(Atomicity, Consistency, Isolation, Durability)とは、**データベーストランザクション**を実現させるための特性である。

![acid.png]({{ "/assets/img/de-glossary/acid.png" | absolute_url }}){: .center-image }{:width="600px"}

### Atomicity(原子性)
トランザクションが中断されると、連携されているデータに何も変更が起きないということ。

### Consistency(一貫性)
トランザクションの前と後のデータの状態（適合性）が変わらないこと。

### Isolation(独立性)
トランザクション中に行われる操作は他のトランザクションに影響を与えない事を保証します。

### Durability(永続性)
ランザクション処理結果は永続的であること。

Atomicityではトランザクションが中断されると結果はデータに反映されないが、Durabilityではトランザクションが完了すると結果がデータに永続的に反映された状態になる。

### References
- [What does the acronym ACID mean?](https://www.quora.com/What-does-the-acronym-ACID-mean)
- [おっさんがACIDとかBASEとかまとめておく。](https://qiita.com/suziq99999/items/2e7037042b31a77b19c8)









