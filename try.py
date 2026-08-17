# 强制UTF-8输出编码
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================
# print(f"长度是 {len('hello')}")
m=0
for i in range(51):
    m+=i+0
print(m)
"""ai如此伟大
"""