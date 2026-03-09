#!/usr/bin/env python3
"""
天文数据下载服务演示
展示各服务的使用方法（离线演示模式）
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class ServiceCapability:
    """服务能力描述"""
    name: str
    category: str  # 'spectrum', 'image', 'lightcurve', 'catalog', 'other'
    description: str
    method: str
    params: Dict[str, str]
    example: str


# 定义所有可用服务
SPECTRUM_SERVICES = [
    ServiceCapability(
        name="SDSS",
        category="spectrum",
        description="斯隆数字巡天光学光谱",
        method="SDSS.query_region() + SDSS.get_spectra()",
        params={"ra": "赤经（度）", "dec": "赤纬（度）", "radius": "搜索半径（角秒）"},
        example="""
from astroquery.sdss import SDSS
spectra = SDSS.query_region(coord, spectro=True, radius=5*u.arcsec)
files = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberid)
        """
    ),
    ServiceCapability(
        name="HEASARC",
        category="spectrum",
        description="高能天体物理X射线光谱",
        method="Heasarc.query_object()",
        params={"target": "目标名称", "mission": "任务名称(xmm/chandra/nustar)"},
        example="""
from astroquery.heasarc import Heasarc
heasarc = Heasarc()
result = heasarc.query_object('Cen X-3', mission='xmmmaster')
        """
    ),
    ServiceCapability(
        name="Splatalogue",
        category="spectrum",
        description="分子光谱线数据库",
        method="Splatalogue.query_lines()",
        params={"min_frequency": "最小频率(GHz)", "max_frequency": "最大频率(GHz)", 
                "chemical_name": "分子名称"},
        example="""
from astroquery.splatalogue import Splatalogue
result = Splatalogue.query_lines(
    min_frequency=100*u.GHz, max_frequency=200*u.GHz, 
    chemical_name='CO'
)
        """
    ),
    ServiceCapability(
        name="NIST",
        category="spectrum",
        description="原子光谱线数据库",
        method="Nist.query()",
        params={"minwav": "最小波长(Å)", "maxwav": "最大波长(Å)", 
                "element": "元素符号", "ion_state": "电离态(0=中性)"},
        example="""
from astroquery.nist import Nist
result = Nist.query(minwav=6500, maxwav=6600, element='H', ion_state=0)
        """
    ),
    ServiceCapability(
        name="IRSA (IRS)",
        category="spectrum",
        description="Spitzer IRS 红外光谱",
        method="Irsa.query_spectra()",
        params={"coord": "天球坐标", "radius": "搜索半径"},
        example="""
from astroquery.irsa import Irsa
result = Irsa.query_spectra(coord=coord, mission='irs')
        """
    ),
    ServiceCapability(
        name="ESO Archive",
        category="spectrum",
        description="欧洲南方台光谱档案",
        method="Eso.query_main()",
        params={"target": "目标名称", "column_filters": "列过滤器"},
        example="""
from astroquery.eso import Eso
eso = Eso()
result = eso.query_main(column_filters={'target': 'Sgr A*'})
        """
    ),
]

IMAGE_SERVICES = [
    ServiceCapability(
        name="SkyView",
        category="image",
        description="多波段巡天图像服务",
        method="SkyView.get_images()",
        params={"position": "天球坐标", "survey": "巡天名称", "radius": "图像半径"},
        example="""
from astroquery.skyview import SkyView
images = SkyView.get_images(
    position=coord, 
    survey=['DSS2 Red', '2MASS-J', 'WISE 3.4'],
    radius=0.1*u.deg
)
        """
    ),
    ServiceCapability(
        name="HiPS2FITS",
        category="image",
        description="HiPS 图像服务",
        method="hips2fits.query()",
        params={"hips": "HiPS标识", "ra": "赤经", "dec": "赤纬", 
                "fov": "视场(度)", "width": "图像宽度(像素)"},
        example="""
from astroquery.hips2fits import hips2fits
result = hips2fits.query(
    hips='CDS/P/DSS2/color',
    ra=10.6847, dec=41.2687,
    fov=0.5, width=500
)
        """
    ),
    ServiceCapability(
        name="ESASky",
        category="image",
        description="ESA 多波段图像服务",
        method="ESASky.query_region_maps()",
        params={"position": "天球坐标", "missions": "任务列表"},
        example="""
from astroquery.esasky import ESASky
maps = ESASky.query_region_maps(
    position=coord, 
    missions=['XMM', 'HERSCHEL']
)
        """
    ),
    ServiceCapability(
        name="ImageCutouts",
        category="image",
        description="巡天图像切片服务",
        method="ImageCutoutsFirst.get_images()",
        params={"coordinates": "坐标", "image_size": "图像大小"},
        example="""
from astroquery.image_cutouts.first import First
images = First.get_images(coord, image_size=1*u.arcmin)
        """
    ),
]

LIGHTCURVE_SERVICES = [
    ServiceCapability(
        name="MAST (TESS/Kepler)",
        category="lightcurve",
        description="空间望远镜光变曲线",
        method="Mast.query_object()",
        params={"target": "目标名称", "obs_collection": "观测集合"},
        example="""
from astroquery.mast import Mast
obs = Mast.query_object(
    'Kepler-186', 
    service="Mast.Caom.Cone",
    params={"obs_collection": "TESS"}
)
        """
    ),
    ServiceCapability(
        name="OGLE",
        category="lightcurve",
        description="光学引力透镜实验光变",
        method="Ogle.query_region()",
        params={"coord": "天球坐标", "width": "搜索宽度"},
        example="""
from astroquery.ogle import Ogle
result = Ogle.query_region(coord=coord, width=0.1*u.deg)
        """
    ),
]

CATALOG_SERVICES = [
    ServiceCapability(
        name="VizieR",
        category="catalog",
        description="星表查询服务",
        method="Vizier.query_region()",
        params={"coordinates": "坐标", "radius": "搜索半径", "catalog": "星表名"},
        example="""
from astroquery.vizier import Vizier
vizier = Vizier(catalog='I/355/gaiadr3')
result = vizier.query_region(coord, radius=1*u.arcmin)
        """
    ),
    ServiceCapability(
        name="SIMBAD",
        category="catalog",
        description="天体识别和分类",
        method="Simbad.query_region()",
        params={"coordinates": "坐标", "radius": "搜索半径"},
        example="""
from astroquery.simbad import Simbad
result = Simbad.query_region(coord, radius=1*u.arcmin)
        """
    ),
    ServiceCapability(
        name="NED",
        category="catalog",
        description="河外天体数据库",
        method="Ned.query_region()",
        params={"coordinates": "坐标", "radius": "搜索半径"},
        example="""
from astroquery.ned import Ned
result = Ned.query_region(coord, radius=0.1*u.deg)
        """
    ),
    ServiceCapability(
        name="Gaia",
        category="catalog",
        description="Gaia DR3 天体测量",
        method="Gaia.query_object()",
        params={"coordinate": "坐标", "radius": "搜索半径"},
        example="""
from astroquery.gaia import Gaia
result = Gaia.query_object_async(coordinate=coord, radius=1*u.arcmin)
        """
    ),
]

OTHER_SERVICES = [
    ServiceCapability(
        name="ALMA",
        category="other",
        description="ALMA 射电干涉数据",
        method="Alma.query()",
        params={"payload": "查询参数"},
        example="""
from astroquery.alma import Alma
result = Alma.query({'source_name_alma': 'Sgr A*'})
        """
    ),
    ServiceCapability(
        name="Exoplanet Archives",
        category="other",
        description="系外行星数据库",
        method="NasaExoplanetArchive.query_object()",
        params={"object_name": "天体名称"},
        example="""
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
result = NasaExoplanetArchive.query_object('Kepler-186')
        """
    ),
]


def print_service_list():
    """打印所有服务列表"""
    all_services = {
        '光谱服务': SPECTRUM_SERVICES,
        '图像服务': IMAGE_SERVICES,
        '光变曲线服务': LIGHTCURVE_SERVICES,
        '星表服务': CATALOG_SERVICES,
        '其他服务': OTHER_SERVICES,
    }
    
    print("=" * 80)
    print("Astroquery 天文数据下载服务一览")
    print("=" * 80)
    
    for category, services in all_services.items():
        print(f"\n{'='*40}")
        print(f"📂 {category}")
        print(f"{'='*40}")
        
        for svc in services:
            print(f"\n  📌 {svc.name}")
            print(f"     描述: {svc.description}")
            print(f"     方法: {svc.method}")
            print(f"     参数: {', '.join(svc.params.keys())}")
    
    print("\n" + "=" * 80)
    print(f"总计: {sum(len(s) for s in all_services.values())} 个服务")
    print("=" * 80)


def print_detailed_examples():
    """打印详细示例"""
    all_services = (
        SPECTRUM_SERVICES + IMAGE_SERVICES + 
        LIGHTCURVE_SERVICES + CATALOG_SERVICES + OTHER_SERVICES
    )
    
    print("\n" + "=" * 80)
    print("详细代码示例")
    print("=" * 80)
    
    # 选择几个代表性的服务
    featured = ["SDSS", "SkyView", "VizieR", "MAST (TESS/Kepler)", "HEASARC"]
    
    for svc in all_services:
        if svc.name in featured:
            print(f"\n{'='*60}")
            print(f"📌 {svc.name} - {svc.description}")
            print(f"{'='*60}")
            print(f"方法: {svc.method}")
            print(f"\n参数说明:")
            for param, desc in svc.params.items():
                print(f"  • {param}: {desc}")
            print(f"\n代码示例:")
            print(svc.example)


def save_to_json():
    """保存服务信息到JSON"""
    all_services = {
        'spectra': [asdict(s) for s in SPECTRUM_SERVICES],
        'images': [asdict(s) for s in IMAGE_SERVICES],
        'lightcurves': [asdict(s) for s in LIGHTCURVE_SERVICES],
        'catalogs': [asdict(s) for s in CATALOG_SERVICES],
        'other': [asdict(s) for s in OTHER_SERVICES],
    }
    
    with open('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astroquery_services.json', 'w', encoding='utf-8') as f:
        json.dump(all_services, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 服务信息已保存到 astroquery_services.json")


if __name__ == "__main__":
    # 打印服务列表
    print_service_list()
    
    # 打印详细示例
    print_detailed_examples()
    
    # 保存到JSON
    save_to_json()
