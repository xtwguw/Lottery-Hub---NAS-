🎱 NAS 彩票数据中心 (Lottery Hub) v2.1 Ultimate
专为家庭 NAS 用户（群晖、飞牛 OS、Unraid 等）设计的轻量级彩票数据管理与分析中心。基于 Docker 容器化部署，支持全量历史数据同步、OCR 智能识图选号、复式计算及智能数据校准。

✨ 核心功能 (v2.1 Ultimate)
🚀 基础架构
全量数据引擎：自动同步双色球、大乐透、七星彩自发行以来所有历史数据（包含红蓝球、奖池、各奖级注数）。

智能调度系统：

自动追号：开奖日晚间自动进入高频轮询模式，第一时间获取开奖结果。

数据校准：每日中午 12:00 自动运行全量校准，确保数据零误差。

系统级优化：基于 SQLite WAL 模式，限制内存占用（<300MB），适合长期后台运行。

💡 交互体验增强
OCR 智能识图：支持手机/电脑上传彩票照片，包含裁剪、旋转、重新选择功能，自动识别号码填入。

开奖实时参照：选号时，下拉框选择历史期号，右侧即时显示该期的开奖号码及日期，方便找规律。

深色模式 (Dark Mode)：支持按日出日落时间自动切换，或手动强制切换，夜间查看不刺眼。

可视化看板：双重倒计时（购票截止/开奖时间），全量历史数据支持年份+期号双重精准筛选与无限滚动加载。

🧮 高级算奖引擎
复式投注计算：完美支持复式（如 8+2、10+3 等）全排列拆分计算。

大乐透追加：支持大乐透追加投注奖金计算。

结果可视化：

命中高亮：中奖号码显示为实心红/蓝球，未中号码为空心球。

未中奖反馈：即使未中奖，也会列出命中情况，拒绝“死得不明不白”。

中文奖金：奖金自动汇总并显示中文大写（如“壹万元整”）。

🛠️ 安装部署指南
1. 准备工作
在您的 NAS 或服务器上创建一个目录，例如 lottery，并在其中创建 templates 子目录：

Bash
mkdir -p lottery/templates
cd lottery
2. 创建文件
请将以下代码分别保存到对应的文件中。

<details> <summary><strong>📄 1. app.py (点击展开复制后端代码)</strong></summary>

Python
import os
import sqlite3
import json
import time
import requests
import urllib3
import itertools
import re
import pytesseract
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from threading import Thread
import traceback

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# --- 配置 ---
PORT = 5088
DB_PATH = '/app/data/lottery.db'
URLS = {
    'ssq': 'https://data.17500.cn/ssq_desc.txt',
    'dlt': 'https://data.17500.cn/dlt2_desc.txt',
    '7xc': 'https://data.17500.cn/7xc_desc.txt'
}

DRAW_RULES = {
    'ssq': {'days': [1, 3, 6], 'draw_time': '21:15', 'stop_time': '20:00'}, 
    'dlt': {'days': [0, 2, 5], 'draw_time': '21:25', 'stop_time': '21:00'}, 
    '7xc': {'days': [1, 4, 6], 'draw_time': '21:25', 'stop_time': '21:00'}  
}

@app.errorhandler(Exception)
def handle_exception(e):
    if "404" in str(e): return jsonify(success=False, message="API endpoint not found"), 404
    print(f"❌ Server Error: {str(e)}", flush=True)
    traceback.print_exc()
    return jsonify(success=False, message=f"Server Error: {str(e)}"), 500

@app.before_request
def log_request_info():
    if '/static/' in request.path: return
    if request.path not in ['/api/init']: 
        ip = request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔌 IP: {ip} -> {request.path}", flush=True)

# ==================== 1. 初始化与路由 ====================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def setup_db_optimization():
    if not os.path.exists('/app/data'): os.makedirs('/app/data')
    try:
        conn = get_db()
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute('''CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, type TEXT, issue TEXT, date TEXT, red TEXT, blue TEXT, prizes TEXT, raw TEXT)''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ti ON history (type, issue)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON history (date)')
        conn.commit(); conn.close()
        print(f"[{datetime.now()}] ✅ DB Optimized (WAL Mode)", flush=True)
    except Exception as e: print(f"❌ DB Init Error: {e}", flush=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init')
def api_init():
    conn = get_db()
    resp = {}
    for t in URLS.keys():
        try:
            rows = conn.execute(f'SELECT issue, date, red, blue, prizes FROM history WHERE type="{t}" ORDER BY date DESC LIMIT 10').fetchall()
            years_data = conn.execute(f'SELECT DISTINCT substr(date, 1, 4) as year FROM history WHERE type="{t}" ORDER BY year DESC').fetchall()
            years = [r['year'] for r in years_data]
            all_issues_data = conn.execute(f'SELECT issue, date, red, blue FROM history WHERE type="{t}" ORDER BY date DESC').fetchall()
            all_issues = [dict(r) for r in all_issues_data]
            latest_issue = rows[0] if rows else None
            resp[t] = {
                'history': [dict(r) for r in rows],
                'years': years,
                'all_issues': all_issues,
                'latest_issue': dict(latest_issue) if latest_issue else None,
                'next_draw': get_next_event_time(t, 'draw_time'),
                'next_stop': get_next_event_time(t, 'stop_time')
            }
        except:
            resp[t] = {'history':[], 'years':[], 'all_issues':[], 'next_draw': '...', 'next_stop': '...'}
    conn.close()
    return jsonify(resp)

@app.route('/api/history_list', methods=['POST'])
def api_history_list():
    data = request.json or {}
    ltype = data.get('type', 'ssq')
    year = data.get('year', 'all')
    issue = data.get('issue', 'all')
    offset = data.get('offset', 0)
    limit = data.get('limit', 20)
    conn = get_db()
    try:
        sql = f'SELECT issue, date, red, blue, prizes FROM history WHERE type="{ltype}"'
        if issue != 'all': sql += f' AND issue="{issue}"'
        elif year != 'all': sql += f' AND date LIKE "{year}%"'
        sql += f' ORDER BY date DESC LIMIT {limit} OFFSET {offset}'
        rows = conn.execute(sql).fetchall()
        return jsonify(success=True, data=[dict(r) for r in rows])
    except Exception as e: return jsonify(success=False, message=str(e)), 500
    finally: conn.close()

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    if 'file' not in request.files: return jsonify({'success': False, 'message': '无文件'}), 400
    file = request.files['file']
    ltype = request.form.get('type', 'ssq')
    try:
        image = preprocess_image(file.stream, ltype)
        whitelist = "0123456789 +-[]"
        custom_config = f'--psm 6 -c tessedit_char_whitelist="{whitelist}"'
        raw_text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        filtered = []
        for l in lines:
            if re.search(r'202\d', l): continue
            if len(re.findall(r'\d', l)) < 5: continue
            filtered.append(l)
        results = []
        if ltype == 'ssq': results = parse_ssq(filtered)
        elif ltype == 'dlt': results = parse_dlt(filtered)
        elif ltype == '7xc': results = parse_7xc(filtered)
        if not results: return jsonify({'success': False, 'message': '未识别到号码，请手动输入'})
        return jsonify({'success': True, 'lines': results})
    except Exception as e: return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/check', methods=['POST'])
def api_check():
    data = request.json or {}
    ltype = data.get('type', 'ssq')
    bets = data.get('bets', [])
    mode = data.get('mode', 'current')
    issue = data.get('issue', 'latest')
    if not isinstance(bets, list): return jsonify([])
    conn = get_db()
    try:
        sql = f"SELECT * FROM history WHERE type='{ltype}'"
        if mode == 'current' and issue != 'latest': sql += f" AND issue='{issue}'"
        sql += " ORDER BY date DESC"
        if mode == 'current' and issue == 'latest': sql += " LIMIT 1"
        draws = conn.execute(sql).fetchall()
    except: draws = []
    finally: conn.close()
    results = []
    for bet in bets:
        if not isinstance(bet, dict) or 'nums' not in bet: continue
        b_str = bet['nums'].strip()
        if not b_str: continue
        matches = []
        for d in draws:
            try:
                res = calc_compound_win(ltype, b_str, d, data.get('zhuijia', False))
                if res['is_win'] or mode == 'current': 
                    res['total_money_cn'] = num_to_chinese(res['total_money']) if res['is_win'] else "未中奖"
                    matches.append({'issue': d['issue'], 'date': d['date'], 'draw_red': d['red'], 'draw_blue': d['blue'], 'win_data': res})
            except: continue
        if matches or mode == 'current': results.append({'bet': b_str, 'matches': matches})
    return jsonify(results)

# === 核心算法部分 (为节省篇幅，省略部分重复函数，请使用完整版代码中的逻辑) ===
def num_to_chinese(money):
    if money == 0: return "零元整"
    units = ['', '拾', '佰', '仟']; big_units = ['', '万', '亿', '兆']; nums = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖']
    money_str = str(int(money)); n = len(money_str)
    if n > 16: return str(money)
    result = []; zero_flag = False
    for i, digit in enumerate(reversed(money_str)):
        unit_idx = i % 4; big_unit_idx = i // 4; num = int(digit)
        if unit_idx == 0 and i > 0:
            if zero_flag: 
                if result and result[-1] == '零': result.pop()
                zero_flag = False
            result.append(big_units[big_unit_idx])
        if num > 0:
            if zero_flag: result.append('零'); zero_flag = False
            result.append(units[unit_idx]); result.append(nums[num])
        else:
            if not zero_flag: zero_flag = True
    final_str = "".join(reversed(result)).replace("亿万", "亿").replace("兆亿", "兆")
    if final_str.endswith("零"): final_str = final_str[:-1]
    return final_str + "元整"

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0: return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    if abs(angle) < 0.5: return image
    (h, w) = image.shape[:2]; center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(image_stream, ltype='ssq'):
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = correct_skew(img)
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ltype == '7xc': kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    else: kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.erode(binary, kernel, iterations=1)
    return Image.fromarray(processed)

def clean_ocr_line(line):
    line = line.replace('o', '0').replace('O', '0').replace('l', '1').replace('z', '2').replace('S', '5')
    line = line.replace('|', ' ').replace('[', ' ').replace(']', ' ')
    return line.strip()

def parse_ssq(lines):
    results = []
    for line in lines:
        line = clean_ocr_line(line).replace('-', ' ')
        nums = re.findall(r'\d{1,2}', line)
        if len(nums) >= 7:
            reds = " ".join([n.zfill(2) for n in nums[:6]])
            blue = nums[6].zfill(2)
            results.append(f"{reds} + {blue}")
    return results

def parse_dlt(lines):
    results = []
    for line in lines:
        line = clean_ocr_line(line)
        if '+' in line:
            parts = line.split('+')
            if len(parts) >= 2:
                front = re.findall(r'\d{1,2}', parts[0])
                back = re.findall(r'\d{1,2}', parts[1])
                if len(front) >= 5 and len(back) >= 2:
                    reds = " ".join([n.zfill(2) for n in front[-5:]])
                    blues = " ".join([n.zfill(2) for n in back[:2]])
                    results.append(f"{reds} + {blues}")
                    continue
        nums = re.findall(r'\d{1,2}', line)
        if len(nums) >= 7:
            reds = " ".join([n.zfill(2) for n in nums[:5]])
            blues = " ".join([n.zfill(2) for n in nums[5:7]])
            results.append(f"{reds} + {blues}")
    return results

def parse_7xc(lines):
    results = []
    for line in lines:
        line = clean_ocr_line(line)
        nums = re.findall(r'\d+', line) 
        if len(nums) >= 7:
            valid_nums = nums[-7:]
            results.append(" ".join(valid_nums))
    return results

def parse_and_save(lot_type, content):
    lines = content.strip().split('\n'); conn = get_db(); count = 0
    for line in lines:
        p = line.split(); 
        if len(p) < 10: continue 
        issue, date = p[0], p[1]; 
        if '-' not in date and len(date) < 8: continue
        prizes = [] 
        try:
            if lot_type == 'ssq':
                red, blue = " ".join(p[2:8]), p[8]
                for i, n in enumerate(['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖']):
                    idx = 17 + i*2; 
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
            elif lot_type == 'dlt':
                red, blue = " ".join(p[2:7]), " ".join(p[7:9])
                for i, n in enumerate(['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖','七等奖','八等奖','九等奖']):
                    idx = 11 + i*2; 
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
                for i, n in enumerate(['一等奖(追加)','二等奖(追加)']):
                    idx = 29 + i*2; 
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
            elif lot_type == '7xc':
                red, blue = " ".join(p[2:9]), ""
                for i, n in enumerate(['特等奖','一等奖','二等奖','三等奖','四等奖','五等奖']):
                    idx = 11 + i*2; 
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
            uid = f"{lot_type}_{issue}"
            conn.execute('INSERT OR REPLACE INTO history (id, type, issue, date, red, blue, prizes, raw) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (uid, lot_type, issue, date, red, blue, json.dumps(prizes, ensure_ascii=False), json.dumps(p)))
            count += 1
        except: continue
    conn.commit(); conn.close(); return count

def sync_data(specific_type=None):
    targets = {specific_type: URLS[specific_type]} if specific_type else URLS
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.17500.cn/"}
    for k, url in targets.items():
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=30); r.encoding = r.apparent_encoding
            if r.status_code == 200: 
                c = parse_and_save(k, r.text)
                if c > 0: print(f"[{datetime.now().strftime('%H:%M')}] ✅ {k} updated {c}", flush=True)
        except Exception as e: print(f"❌ Sync {k} failed: {e}", flush=True)

def smart_scheduler():
    print(f"[{datetime.now()}] 🚀 智能调度已启动", flush=True); last_sync = {}; time.sleep(5); sync_data()
    while True:
        try:
            now = datetime.now(); current_date_str = now.strftime('%Y-%m-%d')
            if now.hour == 12 and now.minute == 0: sync_data(); time.sleep(65); continue
            for ltype, rule in DRAW_RULES.items():
                if now.weekday() in rule['days']:
                    draw_h, draw_m = map(int, rule['draw_time'].split(':'))
                    draw_time = now.replace(hour=draw_h, minute=draw_m, second=0)
                    start_check = draw_time + timedelta(minutes=10); stop_check = draw_time + timedelta(hours=4) 
                    if start_check <= now <= stop_check:
                        if not check_has_today_data(ltype, current_date_str):
                            if time.time() - last_sync.get(ltype, 0) > 300:
                                print(f"🕒 {ltype} tracking...", flush=True); sync_data(ltype); last_sync[ltype] = time.time()
            time.sleep(60)
        except: time.sleep(60)

def check_has_today_data(ltype, date_str):
    try: conn = get_db(); row = conn.execute(f"SELECT id FROM history WHERE type='{ltype}' AND date='{date_str}' LIMIT 1").fetchone(); conn.close(); return row is not None
    except: return False
def get_next_event_time(ltype, time_key):
    now = datetime.now(); rule = DRAW_RULES[ltype]; target_h, target_m = map(int, rule[time_key].split(':'))
    target = now.replace(hour=target_h, minute=target_m, second=0)
    if now <= target and now.weekday() in rule['days']: return target.strftime("%Y-%m-%d %H:%M:%S")
    target += timedelta(days=1)
    while target.weekday() not in rule['days']: target += timedelta(days=1)
    return target.strftime("%Y-%m-%d %H:%M:%S")
def get_combinations(nums, count): return list(itertools.combinations(nums, count))
def check_single_ssq(u_red, u_blue, d_red, d_blue):
    r, b = len(set(u_red) & set(d_red)), 1 if u_blue == d_blue else 0
    if r==6 and b==1: return 1, '一等奖'; 
    if r==6: return 2, '二等奖'; 
    if r==5 and b==1: return 3, '三等奖'
    if r==5 or (r==4 and b==1): return 4, '四等奖'; 
    if r==4 or (r==3 and b==1): return 5, '五等奖'; 
    if b==1: return 6, '六等奖'; return 0, ''
def check_single_dlt(u_f, u_b, d_f, d_b):
    mf, mb = len(set(u_f) & set(d_f)), len(set(u_b) & set(d_b))
    if mf==5 and mb==2: return 1, '一等奖'; 
    if mf==5 and mb==1: return 2, '二等奖'; 
    if mf==5: return 3, '三等奖'
    if mf==4 and mb==2: return 4, '四等奖'; 
    if mf==4 and mb==1: return 5, '五等奖'; 
    if mf==3 and mb==2: return 6, '六等奖'
    if mf==4: return 7, '七等奖'; 
    if (mf==3 and mb==1) or (mf==2 and mb==2): return 8, '八等奖'; 
    if mf==3 or (mf==1 and mb==2) or (mf==2 and mb==1) or (mf==0 and mb==2): return 9, '九等奖'; return 0, ''
def check_single_7xc(u_nums, d_nums):
    hits = sum(1 for i in range(min(len(u_nums), len(d_nums))) if u_nums[i] == d_nums[i])
    if hits == 7: return 1, '特等奖'; 
    if hits == 6: return 2, '一等奖'; 
    if hits >= 4: return 6, '五等奖'; return 0, ''
def calc_compound_win(ltype, bet_nums, draw, is_zj=False):
    d_red = draw['red'].split(); d_blue = draw['blue'].split() if draw['blue'] else []
    if ltype == '7xc': d_blue = []
    u_reds, u_blues, combs_red, combs_blue = [], [], [], []
    try:
        if ltype == 'ssq':
            if '+' in bet_nums: p=bet_nums.split('+'); u_reds=p[0].replace(',',' ').split(); u_blues=p[1].replace(',',' ').split()
            else: raw=bet_nums.replace(',',' ').split(); u_reds=raw[:-1]; u_blues=[raw[-1]] if len(raw)>0 else []
            combs_red, combs_blue = get_combinations(u_reds, 6), get_combinations(u_blues, 1)
        elif ltype == 'dlt':
            if '+' in bet_nums: p=bet_nums.split('+'); u_reds=p[0].replace(',',' ').split(); u_blues=p[1].replace(',',' ').split()
            else: raw=bet_nums.replace(',',' ').split(); u_reds=raw[:5]; u_blues=raw[5:]
            combs_red, combs_blue = get_combinations(u_reds, 5), get_combinations(u_blues, 2)
        elif ltype == '7xc': u_reds=bet_nums.replace(',',' ').split(); combs_red=[u_reds]; combs_blue=[[]]
    except: return {'is_win': False, 'details': {}}
    prize_map = {}; summary, total_money = {}, 0
    try:
        if draw['prizes']:
            for p in json.loads(draw['prizes']): prize_map[p['n']] = float(p['m'].replace(',',''))
    except: pass
    for r_c in combs_red:
        for b_c in combs_blue:
            tier, name = 0, ''
            if ltype == 'ssq': tier, name = check_single_ssq(r_c, b_c[0] if b_c else None, d_red, d_blue[0] if d_blue else None)
            elif ltype == 'dlt': tier, name = check_single_dlt(r_c, b_c, d_red, d_blue)
            elif ltype == '7xc': tier, name = check_single_7xc(r_c, d_red)
            if tier > 0:
                money = prize_map.get(name, 0)
                if ltype == 'dlt' and is_zj and tier <= 2: money += prize_map.get(name+"(追加)", 0)
                if name not in summary: summary[name] = {'count': 0, 'money': 0}
                summary[name]['count'] += 1; summary[name]['money'] += money; total_money += money
    hit_red = list(set(u_reds) & set(d_red)); hit_blue = list(set(u_blues) & set(d_blue))
    return {'total_money': total_money, 'details': summary, 'hit_red': hit_red, 'hit_blue': hit_blue, 'is_win': total_money > 0 or len(summary) > 0}

print(f"[{datetime.now()}] 正在初始化...", flush=True); setup_db_optimization(); Thread(target=smart_scheduler, daemon=True).start()
if __name__ == '__main__': app.run(host='0.0.0.0', port=PORT)
</details>

<details> <summary><strong>📄 2. templates/index.html (点击展开复制前端代码)</strong></summary>

注意：此文件必须放在 templates 子文件夹内。

HTML
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <title>全量彩票数据中心</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css" rel="stylesheet">
    <style>
        /* === 1. 配色系统 === */
        :root {
            --bg-body: #f4f6f9; --bg-card: #ffffff;
            --text-main: #212529; --text-sub: #6c757d; --text-label: #495057;
            --border-color: #dee2e6; --header-bg: #2c3e50; --header-text: #ffffff;
            --input-bg: #ffffff; --ball-bg: #ffffff; --ball-text: #333333; --ball-border: #dee2e6;
            --highlight-bg: #fff8e1; --highlight-border: #ffe0b2; --preview-bg: #ffffff;
            --tab-active-bg: #0d6efd; --tab-active-text: #ffffff;
            --table-head-bg: #f8f9fa;
        }
        [data-theme="dark"] {
            --bg-body: #000000; --bg-card: #1c1c1e;
            --text-main: #ffffff; --text-sub: #cfcfcf; --text-label: #e0e0e0;
            --border-color: #38383a; --header-bg: #1c1c1e; --header-text: #ffffff;
            --input-bg: #2c2c2e; --ball-bg: #2c2c2e; --ball-text: #ffffff; --ball-border: #48484a;
            --highlight-bg: #3a3a3c; --highlight-border: #5a4020; --preview-bg: #2c2c2e;
            --tab-active-bg: #0a84ff; --tab-active-text: #ffffff;
            --table-head-bg: #2c2c2e;
        }

        body { background-color: var(--bg-body); color: var(--text-main); font-family: -apple-system, sans-serif; padding-bottom: 80px; transition: background 0.3s, color 0.3s; }
        .header-bar { background: var(--header-bg); color: var(--header-text); padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .card-header, .card-footer { background-color: var(--bg-card); border-bottom: 1px solid var(--border-color); border-top: 1px solid var(--border-color); color: var(--text-main); font-weight: bold; }
        .form-control, .form-select { background-color: var(--input-bg); border-color: var(--border-color); color: var(--text-main); }
        .form-control:focus, .form-select:focus { background-color: var(--input-bg); color: var(--text-main); border-color: #0d6efd; box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25); }
        .nav-pills .nav-link { color: var(--text-sub); background: var(--bg-body); margin: 0 2px; }
        .nav-pills .nav-link.active { background-color: var(--tab-active-bg); color: var(--tab-active-text); }
        .ball { display: inline-flex; justify-content: center; align-items: center; width: 34px; height: 34px; border-radius: 50%; margin: 3px; font-weight: bold; font-size: 14px; cursor: pointer; border: 1px solid var(--ball-border); background: var(--ball-bg); color: var(--ball-text); transition: transform 0.1s; }
        .ball:active { transform: scale(0.9); }
        .ball.active-red { background-color: #e74c3c; color: white; border-color: #c0392b; }
        .ball.active-blue { background-color: #3498db; color: white; border-color: #2980b9; }
        .static-ball { display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center; border-radius: 50%; margin: 1px; font-size: 12px; color: white; }
        .sb-red { background: #e74c3c; } .sb-blue { background: #3498db; }
        .picker-area { background: var(--bg-card); border-radius: 8px; padding: 15px; border: 1px solid var(--border-color); margin-bottom: 15px; }
        .picker-header-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed var(--border-color); padding-bottom: 8px; margin-bottom: 10px; flex-wrap: wrap; }
        .picker-title { font-size: 1.1rem; font-weight: bold; color: var(--text-main); margin-right: 10px; }
        #draw-info-display { font-size: 0.9rem; color: var(--text-sub); display: flex; align-items: center; }
        .picker-section-title { font-size: 13px; color: var(--text-sub); margin-bottom: 5px; font-weight: bold; }
        .current-pick-display { font-family: monospace; color: #d63384; font-weight: bold; font-size: 1.1em; }
        .qxc-row { display: flex; align-items: center; border-bottom: 1px solid var(--border-color); padding: 5px 0; }
        .qxc-label { width: 50px; font-size: 13px; color: var(--text-label); font-weight: bold; }
        .qxc-balls { flex: 1; display: flex; flex-wrap: wrap; gap: 2px; }
        #ocr-preview-container { border: 2px dashed var(--border-color); border-radius: 8px; padding: 10px; background: var(--preview-bg); height: 100%; min-height: 220px; display: flex; align-items: center; justify-content: center; position: relative; flex-direction: column; }
        #ocr-preview-img { max-width: 100%; max-height: 350px; border-radius: 4px; display: none; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        #clear-img-btn { position: absolute; top: 10px; right: 10px; z-index: 20; background: rgba(255,255,255,0.8); }
        .placeholder-text { color: var(--text-sub); font-size: 0.9rem; text-align: center; }
        .theme-switch-wrapper { display: flex; align-items: center; gap: 8px; }
        .theme-switch { position: relative; display: inline-block; width: 40px; height: 22px; }
        .theme-switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #0d6efd; }
        input:checked + .slider:before { transform: translateX(18px); }
        .history-box { height: 400px; overflow-y: auto; }
        .ocr-btn-group { position: absolute; right: 10px; top: 10px; z-index: 10; display: flex; gap: 8px; }
        .ocr-btn { opacity: 0.9; width: 36px; height: 36px; padding: 0; line-height: 34px; }
        [data-theme="dark"] .text-secondary { color: #b0b0b0 !important; }
        .result-draw-balls { margin-top: 5px; padding-top: 5px; border-top: 1px dashed rgba(0,0,0,0.1); }
        [data-theme="dark"] .result-draw-balls { border-top-color: rgba(255,255,255,0.1); }
        .result-draw-balls .static-ball { width: 22px; height: 22px; line-height: 22px; font-size: 11px; }
        .mini-ball { display: inline-block; width: 20px; height: 20px; line-height: 20px; text-align: center; border-radius: 50%; margin: 0 1px; font-size: 11px; color: white; font-weight: bold; }
    </style>
</head>
<body>

<div class="header-bar">
    <div style="width: 70px;"></div>
    <div class="text-center">
        <h5 class="m-0 fw-bold">全量彩票数据中心</h5>
        <div id="clock" class="small mt-1 opacity-75" style="font-size: 0.75rem;">Loading...</div>
    </div>
    <div class="theme-switch-wrapper">
        <span id="theme-icon" style="font-size:1.2rem;">🌞</span>
        <label class="theme-switch">
            <input type="checkbox" id="checkbox" onchange="toggleTheme(this.checked)">
            <div class="slider"></div>
        </label>
    </div>
</div>

<div class="container">
    <ul class="nav nav-pills nav-fill mb-4 p-1 shadow-sm rounded" style="background-color: var(--bg-card); border: 1px solid var(--border-color);">
        <li class="nav-item"><button id="tab-ssq" class="nav-link active" onclick="loadTab('ssq')">双色球</button></li>
        <li class="nav-item"><button id="tab-dlt" class="nav-link" onclick="loadTab('dlt')">大乐透</button></li>
        <li class="nav-item"><button id="tab-7xc" class="nav-link" onclick="loadTab('7xc')">七星彩</button></li>
    </ul>

    <div class="row g-3 mb-4">
        <div class="col-6"><div class="card h-100"><div class="card-body text-center py-2"><div class="small fw-bold text-secondary">🛑 截止时间</div><div id="cd-stop" class="badge bg-secondary mt-1">--</div></div></div></div>
        <div class="col-6"><div class="card h-100"><div class="card-body text-center py-2"><div class="small fw-bold text-secondary">🎉 开奖时间</div><div id="cd-draw" class="badge bg-danger mt-1">--</div></div></div></div>
    </div>

    <div class="card mb-4">
        <div class="card-header d-flex justify-content-between align-items-center pt-3">
            <span><i class="bi bi-clock-history"></i> 历史开奖</span>
            <div class="d-flex gap-1">
                <select id="filter-year" class="form-select form-select-sm" style="width:auto; min-width:80px;" onchange="resetAndLoadHistory(this.value)"></select>
                <select id="filter-history-issue" class="form-select form-select-sm" style="width:auto; min-width:90px;" onchange="filterHistoryByIssue(this.value)">
                    <option value="all">所有期数</option>
                </select>
            </div>
        </div>
        <div class="history-box p-0" id="history-box">
            <div id="history-list" class="accordion accordion-flush"></div>
            <div id="history-loading" class="text-center p-3 small text-muted" style="display:none">加载更多...</div>
        </div>
    </div>

    <div class="card mb-5">
        <div class="card-header d-flex justify-content-between align-items-center pt-3">
            <span><i class="bi bi-calculator"></i> 选号查询</span>
            <div id="dlt-opt" style="display:none;" class="form-check form-switch m-0"><input class="form-check-input" type="checkbox" id="zj-chk"><label class="form-check-label small">追加</label></div>
        </div>
        <div class="card-body">
            <div class="row g-4">
                <div class="col-lg-7 col-12">
                    <div class="mb-3"><select class="form-select" id="issue-sel" onchange="updateDrawDisplay()"><option value="latest">核对最新一期</option></select></div>
                    
                    <div class="picker-area">
                        <div class="picker-header-row">
                            <div id="picker-title-container"></div>
                            <div id="draw-info-display"></div>
                        </div>
                        <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom" style="border-color:var(--border-color)!important;">
                            <span class="small text-secondary">暂存: <span id="current-pick-text" class="current-pick-display ms-2"></span></span>
                            <button class="btn btn-sm btn-success rounded-pill px-3" onclick="addCurrentPickToList()"><i class="bi bi-plus-lg"></i> 下一组</button>
                        </div>
                        <div id="number-picker"></div>
                    </div>

                    <div class="position-relative mb-3">
                        <textarea class="form-control shadow-none" id="bet-nums" rows="6" placeholder="号码列表 (每行一组)" oninput="syncPickerFromText()"></textarea>
                        <div class="ocr-btn-group">
                            <input type="file" id="file-input" accept="image/*" hidden onchange="initCrop(this)">
                            <button class="btn btn-light border ocr-btn shadow-sm text-primary" onclick="document.getElementById('file-input').click()"><i class="bi bi-image"></i></button>
                            <input type="file" id="camera-input" accept="image/*" capture="environment" hidden onchange="initCrop(this)">
                            <button class="btn btn-light border ocr-btn shadow-sm" onclick="document.getElementById('camera-input').click()"><i class="bi bi-camera-fill"></i></button>
                        </div>
                    </div>

                    <div class="d-flex gap-2">
                        <button class="btn btn-outline-secondary flex-fill" onclick="clearAll()">清空</button>
                        <button class="btn btn-primary flex-fill" onclick="doCheck('current')">查询选中期</button>
                        <button class="btn btn-outline-danger flex-fill" onclick="doCheck('history')">扫历史</button>
                    </div>
                </div>
                
                <div class="col-lg-5 col-12">
                    <div id="ocr-preview-container">
                        <button type="button" class="btn-close" id="clear-img-btn" aria-label="Close" onclick="clearImagePreview()" title="移除图片"></button>
                        <div id="ocr-placeholder" class="placeholder-text">
                            <i class="bi bi-image" style="font-size: 3rem; opacity: 0.3;"></i><br>
                            识别原图将显示在此处<br>方便对比修改
                        </div>
                        <img id="ocr-preview-img" src="" alt="OCR Preview">
                    </div>
                </div>
            </div>
        </div>
        <div id="check-results" class="card-footer border-top-0"></div>
    </div>
</div>

<div class="modal fade" id="cropModal" tabindex="-1" data-bs-backdrop="static">
    <div class="modal-dialog modal-dialog-centered modal-lg">
        <div class="modal-content border-0 shadow" style="background-color: var(--bg-card); color: var(--text-main);">
            <div class="modal-header border-bottom py-2" style="border-color:var(--border-color)!important;"><h6 class="m-0 fw-bold">图片处理</h6><button type="button" class="btn-close" data-bs-dismiss="modal" style="filter: var(--text-main) == '#ffffff' ? invert(1) : none;"></button></div>
            <div class="modal-body bg-black"><div style="height:60vh;background:#000;border-radius:8px;overflow:hidden"><img id="image-to-crop" src="" style="max-width:100%"></div><div class="mt-3 d-flex align-items-center gap-3"><i class="bi bi-arrow-counterclockwise fs-4 text-white" onclick="rotateImage(-90)"></i><input type="range" class="form-range flex-grow-1" min="-45" max="45" value="0" oninput="rotateImage(this.value, true)"><i class="bi bi-arrow-clockwise fs-4 text-white" onclick="rotateImage(90)"></i></div></div>
            <div class="modal-footer border-top py-2" style="border-color:var(--border-color)!important;"><button id="crop-confirm-btn" class="btn btn-primary w-100" onclick="performCropAndUpload()">开始识别</button></div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
<script>
    let currentType = 'ssq', globalData = null, timerInt;
    let selectedBalls = { red: [], blue: [], qxc: [[],[],[],[],[],[],[]] }; 
    let cropper = null, cropModal = null;
    let historyState = { offset: 0, limit: 20, year: 'all', issue: 'all', isLoading: false, hasMore: true };
    let allIssuesMap = {};

    setInterval(() => { document.getElementById('clock').innerText = new Date().toLocaleString('zh-CN'); }, 1000);
    
    function initTheme() {
        const hour = new Date().getHours();
        const isNight = hour >= 18 || hour < 6;
        const saved = localStorage.getItem('theme');
        const isDark = saved === 'dark' || (!saved && isNight);
        updateThemeUI(isDark);
    }
    function toggleTheme(isChecked) { updateThemeUI(isChecked); localStorage.setItem('theme', isChecked ? 'dark' : 'light'); }
    function updateThemeUI(isDark) {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        document.getElementById('checkbox').checked = isDark;
        document.getElementById('theme-icon').innerText = isDark ? "🌙" : "🌞";
        const closeBtns = document.querySelectorAll('.btn-close');
        closeBtns.forEach(btn => btn.style.filter = isDark ? 'invert(1)' : 'none');
    }

    window.onload = function() {
        initTheme();
        if(document.getElementById('cropModal')) cropModal = new bootstrap.Modal(document.getElementById('cropModal'));
        fetch('/api/init').then(r=>r.json()).then(d=>{ 
            globalData=d; loadTab('ssq'); 
            document.getElementById('history-box').addEventListener('scroll', (e) => {
                if(e.target.scrollTop + e.target.clientHeight >= e.target.scrollHeight - 50) loadMoreHistory();
            });
        }).catch(e=>console.error(e));
    };

    function loadTab(type) {
        currentType = type;
        document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
        document.getElementById('tab-' + type).classList.add('active');
        clearAll(); renderPicker(type);
        if (!globalData || !globalData[type]) return;
        const d = globalData[type];
        startDualTimer(d.next_stop, d.next_draw);
        
        const ySel = document.getElementById('filter-year');
        ySel.innerHTML = '<option value="all">所有年份</option>' + d.years.map(y=>`<option value="${y}">${y}年</option>`).join('');
        ySel.value = 'all';
        
        allIssuesMap = {};
        const sel = document.getElementById('issue-sel');
        let issueOpts = `<option value="latest">核对最新一期</option>`;
        d.all_issues.forEach(i => { allIssuesMap[i.issue] = i; issueOpts += `<option value="${i.issue}">第 ${i.issue} 期</option>`; });
        sel.innerHTML = issueOpts;
        if (d.all_issues.length > 0) allIssuesMap['latest'] = d.all_issues[0];
        
        updateDrawDisplay();

        document.getElementById('dlt-opt').style.display = (type === 'dlt') ? 'block' : 'none';
        resetAndLoadHistory('all');
    }

    function updateDrawDisplay() {
        const val = document.getElementById('issue-sel').value;
        const display = document.getElementById('draw-info-display');
        const data = allIssuesMap[val];
        
        if (!data || !data.red) { display.innerHTML = '<span class="badge bg-secondary">待开奖</span>'; return; }
        
        const rArr = data.red.split(' '); 
        const bArr = data.blue ? data.blue.split(' ') : [];
        const balls = rArr.map(n=>`<span class="mini-ball sb-red">${n}</span>`).join('') + 
                      bArr.map(n=>`<span class="mini-ball sb-blue">${n}</span>`).join('');
        
        const infoHtml = `<span class="me-2 small text-secondary">第 ${data.issue} 期 (${data.date})</span>`;
        const labelHtml = `<span class="me-2 small fw-bold">开奖号码:</span>`;
        
        display.innerHTML = `${infoHtml}${labelHtml}${balls}`;
    }

    function renderPicker(type) {
        const container = document.getElementById('number-picker'); container.innerHTML = ''; document.getElementById('current-pick-text').innerText = ''; selectedBalls = { red: [], blue: [], qxc: [[],[],[],[],[],[],[]] };
        let titleText = "";
        if (type === 'ssq') titleText = "双色球选号"; else if (type === 'dlt') titleText = "大乐透选号"; else if (type === '7xc') titleText = "七星彩选号";
        document.getElementById('picker-title-container').innerHTML = `<div class="picker-title">${titleText}</div>`;
        if (type === '7xc') {
            const posNames = ['一','二','三','四','五','六','七(末)'];
            let html = '';
            for(let i=0; i<7; i++) {
                const maxNum = (i === 6) ? 14 : 9;
                let ballsHtml = ''; for(let n=0; n<=maxNum; n++) ballsHtml += `<div class="ball" onclick="toggleBall('qxc', ${i}, '${n}', this)">${n}</div>`;
                const rowStyle = i===6 ? 'background:var(--highlight-bg); border-radius:4px;' : '';
                html += `<div class="qxc-row" style="${rowStyle}"><div class="qxc-label">${posNames[i]}</div><div class="qxc-balls">${ballsHtml}</div></div>`;
            }
            container.innerHTML = html;
        } else {
            const redCount = type === 'ssq' ? 33 : 35; const blueCount = type === 'ssq' ? 16 : 12;
            const redLabel = type === 'dlt' ? '前区 (红球)' : '红球区'; const blueLabel = type === 'dlt' ? '后区 (蓝球)' : '蓝球区';
            let html = `<div class="picker-section-title text-danger">${redLabel}</div><div class="ball-container mb-3">`;
            for(let i=1; i<=redCount; i++) html += `<div class="ball" onclick="toggleBall('red', 0, '${i.toString().padStart(2,'0')}', this)">${i.toString().padStart(2,'0')}</div>`;
            html += `</div><div class="picker-section-title text-primary">${blueLabel}</div><div class="ball-container">`;
            for(let i=1; i<=blueCount; i++) html += `<div class="ball" onclick="toggleBall('blue', 0, '${i.toString().padStart(2,'0')}', this)">${i.toString().padStart(2,'0')}</div>`;
            html += `</div>`;
            container.innerHTML = html;
        }
    }

    function toggleBall(ct, ri, n, el) {
        let ac = (ct === 'red' || (ct === 'qxc' && ri < 6)) ? 'active-red' : 'active-blue'; if (currentType === '7xc' && ri === 6) ac = 'active-blue';
        if (currentType === '7xc') {
            const idx = selectedBalls.qxc[ri].indexOf(n);
            if (idx > -1) { selectedBalls.qxc[ri].splice(idx, 1); el.classList.remove(ac); }
            else { selectedBalls.qxc[ri].push(n); selectedBalls.qxc[ri].sort((a,b)=>Number(a)-Number(b)); el.classList.add(ac); }
        } else {
            const list = selectedBalls[ct]; const idx = list.indexOf(n);
            if (idx > -1) { list.splice(idx, 1); el.classList.remove(ac); }
            else { list.push(n); list.sort(); el.classList.add(ac); }
        }
        updatePreviewText();
    }
    function updatePreviewText(){
        let t=''; if(currentType==='7xc'){ let p=[]; for(let i=0;i<7;i++) p.push(selectedBalls.qxc[i].length?selectedBalls.qxc[i].join(','):'?'); t=p.join(' '); }
        else { t=`${selectedBalls.red.join(' ')}${selectedBalls.blue.length?' + '+selectedBalls.blue.join(' '):''}`; }
        document.getElementById('current-pick-text').innerText=t;
    }
    function addCurrentPickToList(){
        const p=document.getElementById('current-pick-text').innerText.trim(); if(!p||p.includes('?')) return alert("号码不完整");
        const ta=document.getElementById('bet-nums'); ta.value=ta.value.trim()?(ta.value.trim()+'\n'+p):p; ta.scrollTop=ta.scrollHeight;
        document.querySelectorAll('.ball').forEach(b=>b.classList.remove('active-red','active-blue')); selectedBalls={red:[],blue:[],qxc:[[],[],[],[],[],[],[]]}; document.getElementById('current-pick-text').innerText='';
    }
    
    function clearImagePreview() {
        document.getElementById('ocr-preview-img').src = ''; document.getElementById('ocr-preview-img').style.display = 'none';
        document.getElementById('clear-img-btn').style.display = 'none'; document.getElementById('ocr-placeholder').style.display = 'block';
        document.getElementById('file-input').value = ''; document.getElementById('camera-input').value = '';
    }
    function clearAll() { document.getElementById('bet-nums').value = ''; document.getElementById('check-results').innerHTML = ''; clearImagePreview(); document.querySelectorAll('.ball').forEach(b => b.classList.remove('active-red', 'active-blue')); selectedBalls = { red: [], blue: [], qxc: [[],[],[],[],[],[],[]] }; document.getElementById('current-pick-text').innerText = ''; }

    function resetAndLoadHistory(year) { 
        historyState = { offset: 0, limit: 20, year: year, issue: 'all', isLoading: false, hasMore: true }; 
        document.getElementById('history-list').innerHTML = ''; 
        const issueSel = document.getElementById('filter-history-issue');
        const all = globalData[currentType].all_issues;
        let filtered = all;
        if (year !== 'all') filtered = all.filter(i => i.date.startsWith(year));
        issueSel.innerHTML = '<option value="all">所有期数</option>' + filtered.map(i => `<option value="${i.issue}">${i.issue}期</option>`).join('');
        issueSel.value = 'all';
        loadMoreHistory(); 
    }
    function filterHistoryByIssue(issue) {
        historyState = { offset: 0, limit: 1, year: 'all', issue: issue, isLoading: false, hasMore: false };
        document.getElementById('history-list').innerHTML = '';
        loadMoreHistory();
    }
    async function loadMoreHistory() {
        if (historyState.isLoading || (!historyState.hasMore && historyState.issue === 'all')) return;
        historyState.isLoading = true; document.getElementById('history-loading').style.display = 'block';
        try {
            const resp = await fetch('/api/history_list', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ type: currentType, year: historyState.year, issue: historyState.issue, offset: historyState.offset, limit: historyState.limit }) });
            const data = await resp.json();
            if (data.success && data.data.length > 0) {
                renderHistoryAppend(data.data); historyState.offset += data.data.length;
                if (data.data.length < historyState.limit) historyState.hasMore = false;
            } else historyState.hasMore = false;
        } catch(e) { console.error(e); } finally { historyState.isLoading = false; document.getElementById('history-loading').style.display = 'none'; }
    }
    
    function renderHistoryAppend(list) {
        const h = list.map(row => {
            const rArr = row.red.split(' '); const bArr = row.blue ? row.blue.split(' ') : [];
            const ballHtml = rArr.map(n => `<span class="static-ball sb-red">${n}</span>`).join('') + bArr.map(n => `<span class="static-ball sb-blue">${n}</span>`).join('');
            let prizes = ''; 
            try{ 
                const p=JSON.parse(row.prizes); 
                if(p.length) prizes=`<div class="mt-2"><table class="table table-sm table-bordered text-center small mb-0" style="color:var(--text-main); border-color:var(--border-color);"><thead style="background-color:var(--table-head-bg);"><tr><th>奖项</th><th>注数</th><th>奖金</th></tr></thead><tbody>`+p.map(x=>`<tr><td>${x.n}</td><td>${x.c}</td><td class="text-danger fw-bold">${x.m}</td></tr>`).join('')+`</tbody></table></div>`; 
            }catch(e){}
            return `<div class="accordion-item" style="background:transparent;"><h2 class="accordion-header"><button class="accordion-button collapsed py-2" type="button" data-bs-toggle="collapse" data-bs-target="#h-${row.issue}" style="background:var(--bg-card); color:var(--text-main);"><div class="w-100"><div class="d-flex justify-content-between small mb-1" style="color:var(--text-sub);"><span>第 ${row.issue} 期</span><span>${row.date}</span></div><div>${ballHtml}</div></div></button></h2><div id="h-${row.issue}" class="accordion-collapse collapse" data-bs-parent="#history-list"><div class="accordion-body p-2" style="background:var(--bg-body); color:var(--text-main);">${prizes}</div></div></div>`;
        }).join('');
        document.getElementById('history-list').insertAdjacentHTML('beforeend', h);
    }

    function initCrop(el) { const f=el.files[0]; if(!f)return; const r=new FileReader(); r.onload=(e)=>{ document.getElementById('image-to-crop').src=e.target.result; cropModal.show(); document.getElementById('crop-confirm-btn').disabled=false; document.getElementById('crop-confirm-btn').innerText="开始识别"; setTimeout(()=>{if(cropper)cropper.destroy();cropper=new Cropper(document.getElementById('image-to-crop'),{viewMode:1,dragMode:'move',autoCropArea:0.9})},200); }; r.readAsDataURL(f); }
    function rotateImage(v,abs){ if(cropper) abs?cropper.rotateTo(parseInt(v)):cropper.rotate(parseInt(v)); }
    function performCropAndUpload() { if(!cropper)return; const btn=document.getElementById('crop-confirm-btn'); btn.disabled=true; btn.innerText="处理中..."; cropper.getCroppedCanvas({maxWidth:2048,maxHeight:2048,fillColor:'#fff'}).toBlob((blob)=>{const url=URL.createObjectURL(blob);document.getElementById('ocr-preview-img').src=url;document.getElementById('ocr-preview-img').style.display='block';document.getElementById('ocr-placeholder').style.display='none';document.getElementById('clear-img-btn').style.display='block';cropModal.hide();uploadBlob(blob)},'image/jpeg',0.85); }
    async function uploadBlob(blob){ const fd=new FormData(); fd.append('file',blob,'ocr.jpg'); fd.append('type',currentType); try{ const r=await fetch('/api/ocr',{method:'POST',body:fd}); const d=await r.json(); if(d.success){const ta=document.getElementById('bet-nums');ta.value=ta.value.trim()?(ta.value.trim()+'\n'+d.lines.join('\n')):d.lines.join('\n');document.querySelectorAll('.ball').forEach(b=>b.classList.remove('active-red','active-blue'));}else alert(d.message); }catch(e){alert(e.message);} }
    function startDualTimer(st,dt){ clearInterval(timerInt); const cf=(t)=>{const d=new Date(t)-new Date();if(d<0)return "已结束";return `${Math.floor(d/86400000)}天 ${Math.floor((d%86400000)/3600000)}时 ${Math.floor((d%3600000)/60000)}分 ${Math.floor((d%60000)/1000)}秒`}; timerInt=setInterval(()=>{document.getElementById('cd-stop').innerText=cf(new Date(st));document.getElementById('cd-draw').innerText=cf(new Date(dt))},1000); }
    function syncPickerFromText(){document.querySelectorAll('.ball').forEach(b=>b.classList.remove('active-red','active-blue')); selectedBalls={red:[],blue:[],qxc:[[],[],[],[],[],[],[]]};}

    async function doCheck(mode) {
        const raw = document.getElementById('bet-nums').value.trim(); if (!raw) return alert("请输入号码");
        const resDiv = document.getElementById('check-results'); resDiv.innerHTML = '<div class="p-3 text-center text-secondary"><div class="spinner-border spinner-border-sm"></div> 计算中...</div>';
        const bets = raw.split('\n').map(s=>({nums:s})).filter(x=>x.nums);
        try {
            const resp = await fetch('/api/check', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ type:currentType, mode:mode, issue:document.getElementById('issue-sel').value, zhuijia:document.getElementById('zj-chk').checked, bets:bets }) });
            const data = await resp.json();
            if(!Array.isArray(data)) throw new Error(data.message||"Error");
            if(data.length===0) { resDiv.innerHTML='<div class="p-3 text-center text-muted">未中奖</div>'; return; }
            let html = '';
            data.forEach(item => {
                let matchesHtml = '';
                if(item.matches.length === 0) matchesHtml = `<div class="text-secondary small">未中奖</div>`;
                else {
                    item.matches.forEach(m => {
                        const drawR = m.draw_red.split(' ').map(n=>`<span class="static-ball sb-red">${n}</span>`).join('');
                        const drawB = m.draw_blue ? m.draw_blue.split(' ').map(n=>`<span class="static-ball sb-blue">${n}</span>`).join('') : '';
                        const drawBalls = `<div class="result-draw-balls d-flex align-items-center mb-1"><span class="small me-2" style="color:var(--text-sub)">开奖:</span>${drawR}${drawB}</div>`;
                        
                        let titleHtml = '';
                        if (m.win_data.is_win) {
                            titleHtml = `<span class="text-danger fw-bold">¥${m.win_data.total_money.toLocaleString()}</span> <span class="badge bg-warning text-dark">${m.win_data.total_money_cn||''}</span>`;
                        } else {
                            titleHtml = `<span class="badge bg-secondary">未中奖</span>`;
                        }
                        let balls = `<div class="mb-1 small">第${m.issue}期 <span class="text-secondary">(${m.date})</span>: ${titleHtml}</div>`;
                        
                        let details = ''; 
                        if(m.win_data.is_win) {
                            for(let k in m.win_data.details) details += `<span class="badge bg-danger me-1">${k} x${m.win_data.details[k].count}</span>`;
                        }

                        let userBalls = '';
                        let rp=[], bp=[]; 
                        const betStr = item.bet.replace(/[,，]/g, ' ').trim();
                        if (currentType === '7xc') { rp = betStr.split(/\s+/); } 
                        else {
                            if (item.bet.includes('+')) { const parts = item.bet.split('+'); rp = parts[0].replace(/[,，]/g, ' ').trim().split(/\s+/); bp = parts[1].replace(/[,，]/g, ' ').trim().split(/\s+/); } 
                            else { const all = betStr.split(/\s+/); if(currentType==='ssq'){ rp = all.slice(0, all.length-1); bp = all.slice(all.length-1); } else { rp = all.slice(0, all.length-2); bp = all.slice(all.length-2); } }
                        }
                        userBalls = '<div class="mb-2">';
                        rp.forEach(n => { userBalls += `<span class="ball ${m.win_data.hit_red.includes(n) ? 'active-red' : ''}">${n}</span>`; });
                        if (bp.length > 0) { userBalls += ' <span class="text-muted small">+</span> '; bp.forEach(n => { userBalls += `<span class="ball ${m.win_data.hit_blue.includes(n) ? 'active-blue' : ''}">${n}</span>`; }); }
                        userBalls += '</div>';

                        matchesHtml += `<div class="p-2 rounded mb-2 border" style="background-color:var(--highlight-bg); border-color:var(--highlight-border);">${balls}${drawBalls}<hr class="my-2" style="border-style:dashed; opacity:0.3;">${userBalls}${details}</div>`;
                    });
                }
                html += `<div class="p-3 border-bottom" style="border-color:var(--border-color)!important">${matchesHtml}</div>`;
            });
            resDiv.innerHTML = html;
        } catch(e) { resDiv.innerHTML = `<div class="p-3 text-danger">${e.message}</div>`; }
    }
</script>
</body>
</html>
</details>

<details> <summary><strong>📄 3. docker-compose.yml (点击展开)</strong></summary>

YAML
version: '3'
services:
  lottery-web:
    build: .
    image: lottery-web:latest
    container_name: lottery_helper
    restart: always
    ports:
      - "5088:5088"
    dns:
      - 223.5.5.5
      - 114.114.114.114
    mem_limit: 300m
    volumes:
      - ./data:/app/data
    environment:
      - TZ=Asia/Shanghai
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
</details>

<details> <summary><strong>📄 4. Dockerfile (点击展开)</strong></summary>

Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装 OpenCV 和 OCR 系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 5088

CMD ["gunicorn", "-w", "1", "--threads", "50", "-b", "0.0.0.0:5088", "--timeout", "60", "--access-logfile", "-", "app:app"]
</details>

<details> <summary><strong>📄 5. requirements.txt (点击展开)</strong></summary>

Plaintext
flask
requests
gunicorn
opencv-python-headless
Pillow
pytesseract
numpy
</details>

🛠️ 第三部分：部署与使用指南
3.1 停止旧环境
如果您之前部署过，请务必执行以下命令清理旧环境，以避免缓存问题：

Bash
cd /volume1/docker/lottery  # 进入您的目录
docker compose down         # 停止容器
docker image prune -a -f    # 清理旧镜像
3.2 文件上传
确保所有 5 个文件都已覆盖上传到 NAS 的 /lottery 文件夹中。 ⚠️ 重要：index.html 必须放在 templates 文件夹内。

3.3 重新启动
Bash
sudo docker compose up -d --build
等待几分钟，直到看到日志显示 ✅ DB Optimized (WAL Mode)。

📖 第四部分：功能验证手册
历史记录：

在“历史开奖”卡片中，点击左侧下拉框选择“2023年”，右侧下拉框选择“2023005期”，列表应立即刷新并只显示这一条。

点击列表右侧的箭头，展开应能看到详细的“奖项、注数、奖金”表格。

选号查询：

在“选号查询”卡片中，点击下拉框选择某一期历史记录。

验证：下拉框右侧应立即显示出该期的开奖号码和日期。

随便输入一组没中奖的号码，点击“查询选中期”。

验证：结果区域应显示灰色“未中奖”标签，但下方的号码球中，您猜对的数字应被高亮显示（实心红/蓝）。

OCR 识图：

点击相机图标上传图片。

验证：应弹出全屏裁剪界面，且右上角有“X”按钮。识别成功后，图片预览区右上角应有删除按钮，点击可清除图片并重置状态。

现在，您拥有了一个功能极其强大且细节完善的私人彩票数据中心！祝您使用愉快！
