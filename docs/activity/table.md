| 番号 | アクティビティ名               | 前提アクティビティ | 入力                             | 出力                                 | 備考                                           |
| ---- | ------------------------------ | ------------------ | -------------------------------- | ------------------------------------ | ---------------------------------------------- |
| #0   | TF push                        |                    |                                  | TF                                   |                                                |
| #1   | 画像取得                       |                    |                                  | RGB画像, Depth画像                   |                                                |
| #2   | 画像アライン                   | #1                 | Deptth画像、カメラパラメータ     | Aligned Depth画像                    | タイムスタンプは入力と同じにする               |
| #2x  | (深度フィルタリング)           | #2                 | RGB画像, Aligned Depth画像       | Filtered RGB画像, Filtered Depth画像 |                                                |
| #3   | インスタンスセグメンテーション | #1 (#2x)           | RGB画像                          | 各インスタンスのマスク               |                                                |
| #4   | 中心点検出                     | #2, #3             | 各インスタンスのマスク、深度画像 | (u,v,d)                              |                                                |
| #5   | 方向ベクトル検出               | #4                 | (u,v,d)                          | 方向ベクトル                         |                                                |
| #6   | ベースへの座標変換             | #0, #4, #5         | TF                               | (x, y, z, r, p, y)                   | 座標と姿勢の結合も行う                         |
| #7   | rotated_bboxの算出             | #3                 | 各インスタンスのマスク           |                                      |                                                |
| #8   | 長辺・短辺検出                 | #8                 | rotated_bbox                     | long_radius, short_radius            | ModelShapeみたいな抽象メッセージとして扱いたい |
| #9   | オブジェクト位置形状push       | #7, #9             |                                  |                                      |                                                |
| #10  | 把持候補検出                   | #3                 |                                  |                                      |                                                |
| #11  | 把持候補フィルタリング         | #11                |                                  |                                      | アスペクト比                                   |
| #12  | 把持候補選択                   | #12                |                                  |                                      |                                                |
| #13  | ベースへの座標変換             | #0, #13            | TF                               |                                      |                                                |
| #14  | 把持位置push                   | #14                |                                  |                                      |                                                |