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
