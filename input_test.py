import json

# 表结构定义（用于拼接字段名）
table_columns = {
    "graph_tag_idcard_base": [
        "source", "source_name", "etl_time", "batch", "gmsfhm", "xm", "lxdh", "hh",
        "yhzgxdm", "yhzgxdm_detail", "cym", "xbdm", "xbdm_detail", "mzdm", "mzdm_detail",
        "csrq", "xldm", "xldm_detail", "hyzkdm", "hyzkdm_detail", "cyzk_bm", "cyzk_mc",
        "cyzk_zylbdm", "cyzk_zy", "zjxydm", "zjxydm_detail", "byzkdm", "byzkdm_detail",
        "sg", "xxdm", "xxdm_detail", "swrq", "rkglswyydm", "rkglswyydm_detail",
        "jg_ssxqdm", "jg_ssxqdm_detail", "jg_qhnxxdz", "hjdz_ssxqdm", "hjdz_ssxqdm_detail",
        "hjdz_qhnxxdz", "cyjmsfzqk_qfjg_gajgmc", "cyjmsfzqk_yxqqsrq", "cyjmsfzqk_yxqjzrq",
        "sjgsdwdm", "sjgsdwmc", "qrrq", "qryy", "qryy_detail", "lzdq_ssxqdm",
        "lzdq_ssxqdm_detail", "lzdq_qhnxxdz", "qcrq", "qcyy", "qcyy_detail", "qcdq_ssxqdm",
        "qcdq_ssxqdm_detail", "qcdq_qhnxxdz", "rkzt", "add_time"
    ],
    "rzx_wz_wldt_logs": [
        "account", "device_id", "ip", "mac", "url", "app_name", "nickname", "post_time", "content", "label", "protocol"
    ],
    "res_0_clda_jdc": [
        "plate", "brand", "model", "engine_no", "vin", "owner", "displacement", "reg_date", "use_type", "status", "insure_info"
    ],
    "res_zh_gj_info": [
        "person_id", "phone", "start_location", "end_location", "time", "plate", "track_type", "track_status"
    ]
}

# 关键词匹配映射
table_keywords = {
    "graph_tag_idcard_base": ["户籍", "人员", "身份证", "姓名", "民族", "性别", "出生", "学历", "电话", "户主", "人口","男性","女性","迁入","迁出","婚","家庭","宗教","身高","档案","来源","系统","年龄","血型","同步","籍贯","姓","兵役","户口","行业"],
    "rzx_wz_wldt_logs": ["网络", "轨迹", "账号", "内容", "发帖", "app", "ip", "mac", "url", "标签", "协议", "昵称", "设备", "基站","网站","应用","APP","操作","平台","通信","微信","微博","网","用户"],
    "res_0_clda_jdc": ["车辆", "车牌", "机动车", "品牌", "排量", "发动机", "销售", "车辆识别代号", "驾驶", "保险", "状态","汽车","车"],
    "res_zh_gj_info": ["人员", "轨迹", "出发地", "到达地", "车牌", "手机号", "具体位置", "行程", "轨迹状态", "轨迹类型","出行","到","地点","出现"]
}

# 读取 test.json
with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 匹配表并构造 input 字段
def match_tables_with_input(nl_text):
    inputs = []
    for table, keywords in table_keywords.items():
        for kw in keywords:
            if kw in nl_text:
                columns = table_columns[table]
                input_str = f"{table}(" + ", ".join(columns) + ")"
                inputs.append(input_str)
                break
    return " ".join(inputs)  # 如果多表，拼接为一个字符串（以空格分隔）

# 构造新结构
new_data = []
for entry in data:
    new_entry = {
        "id": entry["id"],
        "level": entry.get("level", "未知"),
        "instruction": entry.get("NL", ""),
        "input": match_tables_with_input(entry.get("NL", ""))
    }
    new_data.append(new_entry)

# 写入新文件
with open("test_with_input.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print("✅ 成功生成 test_transformed.json 文件！")
