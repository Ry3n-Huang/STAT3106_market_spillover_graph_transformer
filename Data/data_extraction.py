import pandas as pd

# 1. 读取你刚才下载的成分股 CSV 文件 (记得把里面的文件名换成你真实下载的那个)
# 如果你下载的是 zip，直接把后缀写成 .zip，Pandas 能直接读，都不用你解压！
df = pd.read_csv('CRSP_S&P 500 Constituents.csv') 

# 2. 自动识别列名（WRDS 有时候给大写 PERMNO，有时候给小写 permno）
permno_col = 'PERMNO' if 'PERMNO' in df.columns else 'permno'

# 3. 核心清洗：提取列 -> 剔除空值 -> 去重 -> 转换为纯整数
# 必须转成 int，不然 Pandas 有时候会带上小数点 (比如 10104.0)
unique_permnos = df[permno_col].dropna().drop_duplicates().astype(int)

# 4. 导出为纯文本文件，不要表头 (header=False)，不要行号 (index=False)
unique_permnos.to_csv('permno_list.txt', index=False, header=False)

print(f"搞定！一共提取了 {len(unique_permnos)} 个独一无二的 PERMNO。")