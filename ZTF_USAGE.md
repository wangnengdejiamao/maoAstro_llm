# ZTF 数据下载（无需 ztfquery）

## 新方法：使用 wget 从 IRSA 下载

### 命令行示例

```bash
# 下载 EV UMa 的 ZTF 数据
wget -O ztf_evuma.tbl "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?catalog=ztf_objects_dr24&spatial=cone&objstr=13.1316+53.8585&radius=10&outfmt=1"

# 下载 Sirius 的 ZTF 数据  
wget -O ztf_sirius.tbl "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?catalog=ztf_objects_dr24&spatial=cone&objstr=101.2875+-16.7161&radius=10&outfmt=1"
```

### Python 代码示例

```python
import subprocess
import tempfile

def download_ztf_wget(ra, dec, radius=10):
    """使用 wget 下载 ZTF 数据"""
    
    temp_file = tempfile.mktemp(suffix='.tbl')
    
    url = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
    params = f"catalog=ztf_objects_dr24&spatial=cone&objstr={ra}+{dec}&radius={radius}&outfmt=1"
    
    cmd = ['wget', '-q', '-O', temp_file, f"{url}?{params}"]
    subprocess.run(cmd, timeout=60)
    
    # 解析 IPAC 表格格式
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    
    # 提取数据行
    data_lines = []
    header_found = False
    for line in lines:
        if line.startswith('|'):
            header_found = True
            continue
        if header_found and line.strip() and not line.startswith('\\'):
            data_lines.append(line.strip())
    
    return data_lines

# 使用示例
data = download_ztf_wget(13.1316, 53.8585)
print(f"找到 {len(data)} 个 ZTF 源")
```

### 数据格式说明

IRSA 返回 IPAC 表格格式，包含以下列：
- `oid`: ZTF 源ID
- `ra`, `dec`: 坐标
- `filtercode`: 滤镜 (zg, zr, zi)
- `nobs`: 观测次数
- `medianmag`: 中值星等
- `ngoodobs`: 有效观测数

### 使用修复后的分析脚本

```bash
python ultimate_analysis_fixed.py --ra 13.1316 --dec 53.8585 --name "EV_UMa"
```

输出示例：
```
【7/12】ZTF 光变曲线...
    使用 wget 从 IRSA 下载 ZTF DR24 数据...
    ✓ 找到 16 个 ZTF 源
      - 1775314300046232: zi, nobs=0, mag=N/A
      - 774308400047058: zi, nobs=0, mag=N/A
      - 774208400035733: zr, nobs=2, mag=21.52
      - 773105100036156: zg, nobs=1, mag=21.44
      - 1775114300027350: zg, nobs=0, mag=N/A
```
