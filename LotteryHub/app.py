import os
import sqlite3
import json
import time
import requests
import urllib3
import itertools
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from threading import Thread

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

# 规则配置
DRAW_RULES = {
    'ssq': {'days': [1, 3, 6], 'draw_time': '21:15', 'stop_time': '20:00'}, 
    'dlt': {'days': [0, 2, 5], 'draw_time': '21:25', 'stop_time': '21:00'}, 
    '7xc': {'days': [1, 4, 6], 'draw_time': '21:25', 'stop_time': '21:00'}  
}

# 全局变量记录状态
SCHEDULER_STATUS = {
    "last_run": "从未运行",
    "next_check": "初始化中",
    "is_alive": False
}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库"""
    if not os.path.exists('/app/data'): os.makedirs('/app/data')
    
    # 每次启动尝试清除旧库以应用新规则(可选)
    if os.path.exists(DB_PATH):
        try:
            # 简单判断是否需要重建（此处强制重建以保数据准确）
            # os.remove(DB_PATH) # 如果不想每次重启都重下，注释掉这行
            pass
        except: pass

    try:
        conn = get_db()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY, type TEXT, issue TEXT, date TEXT, 
                red TEXT, blue TEXT, prizes TEXT, raw TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ti ON history (type, issue)')
        conn.commit()
        conn.close()
        print(f"[{datetime.now()}] ✅ 数据库检查完毕", flush=True)
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 数据库错误: {e}", flush=True)

def parse_and_save(lot_type, content):
    lines = content.strip().split('\n')
    conn = get_db()
    count = 0
    for line in lines:
        p = line.split()
        if len(p) < 10: continue 
        issue, date = p[0], p[1]
        if '-' not in date and len(date) < 8: continue
        prizes = [] 
        try:
            if lot_type == 'ssq':
                red, blue = " ".join(p[2:8]), p[8]
                names = ['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖']
                for i, n in enumerate(names):
                    idx = 17 + i*2
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
            elif lot_type == 'dlt':
                red, blue = " ".join(p[2:7]), " ".join(p[7:9])
                base_names = ['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖','七等奖','八等奖','九等奖']
                for i, n in enumerate(base_names):
                    idx = 11 + i*2
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
                zj_names = ['一等奖(追加)','二等奖(追加)']
                for i, n in enumerate(zj_names):
                    idx = 29 + i*2
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})
            elif lot_type == '7xc':
                red, blue = " ".join(p[2:9]), ""
                names = ['特等奖','一等奖','二等奖','三等奖','四等奖','五等奖']
                for i, n in enumerate(names):
                    idx = 11 + i*2
                    if idx+1 < len(p): prizes.append({'n':n, 'c':p[idx], 'm':p[idx+1]})

            uid = f"{lot_type}_{issue}"
            conn.execute('''
                INSERT OR REPLACE INTO history (id, type, issue, date, red, blue, prizes, raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (uid, lot_type, issue, date, red, blue, json.dumps(prizes, ensure_ascii=False), json.dumps(p)))
            count += 1
        except: continue
    conn.commit()
    conn.close()
    return count

def sync_data(specific_type=None):
    targets = {specific_type: URLS[specific_type]} if specific_type else URLS
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始同步...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.17500.cn/"}
    
    updated = False
    for k, url in targets.items():
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=30)
            r.encoding = r.apparent_encoding
            if r.status_code == 200:
                c = parse_and_save(k, r.text)
                if c > 0: 
                    print(f"   -> ✅ {k} 更新 {c} 条", flush=True)
                    updated = True
        except Exception as e:
            print(f"   -> ❌ {k} 失败: {e}", flush=True)
    
    SCHEDULER_STATUS["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return updated

def smart_scheduler():
    """智能调度线程"""
    print(f"[{datetime.now()}] 🚀 智能调度器已启动", flush=True)
    SCHEDULER_STATUS["is_alive"] = True
    last_sync = {} 
    
    # 启动时先同步一次
    time.sleep(5)
    sync_data()
    
    while True:
        try:
            now = datetime.now()
            current_date_str = now.strftime('%Y-%m-%d')
            SCHEDULER_STATUS["next_check"] = now.strftime("%H:%M:%S") + " (运行中)"
            
            # 策略1：每日 12:00 全量补全
            if now.hour == 12 and now.minute == 0:
                print("🕛 执行午间全量更新...")
                sync_data()
                time.sleep(65)
                continue

            # 策略2：开奖日晚间追号
            for ltype, rule in DRAW_RULES.items():
                if now.weekday() in rule['days']:
                    draw_h, draw_m = map(int, rule['draw_time'].split(':'))
                    draw_time = now.replace(hour=draw_h, minute=draw_m, second=0)
                    start_check = draw_time + timedelta(minutes=10)
                    stop_check = draw_time + timedelta(hours=4) 
                    
                    if start_check <= now <= stop_check:
                        # 检查今日数据是否存在
                        if not check_has_today_data(ltype, current_date_str):
                            # 没数据，且距离上次查超过5分钟
                            if time.time() - last_sync.get(ltype, 0) > 300:
                                print(f"🕒 {ltype} 追号同步中...", flush=True)
                                sync_data(ltype)
                                last_sync[ltype] = time.time()
            
            time.sleep(60)
        except Exception as e:
            print(f"⚠️ 调度器异常: {e}", flush=True)
            time.sleep(60)

def check_has_today_data(ltype, date_str):
    try:
        conn = get_db()
        row = conn.execute(f"SELECT id FROM history WHERE type='{ltype}' AND date='{date_str}' LIMIT 1").fetchone()
        conn.close()
        return row is not None
    except: return False

def get_next_event_time(ltype, time_key):
    now = datetime.now()
    rule = DRAW_RULES[ltype]
    target_h, target_m = map(int, rule[time_key].split(':'))
    target = now.replace(hour=target_h, minute=target_m, second=0)
    if now <= target and now.weekday() in rule['days']: return target.strftime("%Y-%m-%d %H:%M:%S")
    target += timedelta(days=1)
    while target.weekday() not in rule['days']: target += timedelta(days=1)
    return target.strftime("%Y-%m-%d %H:%M:%S")

# --- 算奖逻辑区域 ---
def get_combinations(nums, count): return list(itertools.combinations(nums, count))
def check_single_ssq(u_red, u_blue, d_red, d_blue):
    r, b = len(set(u_red) & set(d_red)), 1 if u_blue == d_blue else 0
    if r==6 and b==1: return 1, '一等奖'
    if r==6: return 2, '二等奖'
    if r==5 and b==1: return 3, '三等奖'
    if r==5 or (r==4 and b==1): return 4, '四等奖'
    if r==4 or (r==3 and b==1): return 5, '五等奖'
    if b==1: return 6, '六等奖'
    return 0, ''
def check_single_dlt(u_f, u_b, d_f, d_b):
    mf, mb = len(set(u_f) & set(d_f)), len(set(u_b) & set(d_b))
    if mf==5 and mb==2: return 1, '一等奖'
    if mf==5 and mb==1: return 2, '二等奖'
    if mf==5: return 3, '三等奖'
    if mf==4 and mb==2: return 4, '四等奖'
    if mf==4 and mb==1: return 5, '五等奖'
    if mf==3 and mb==2: return 6, '六等奖'
    if mf==4: return 7, '七等奖'
    if (mf==3 and mb==1) or (mf==2 and mb==2): return 8, '八等奖'
    if mf==3 or (mf==1 and mb==2) or (mf==2 and mb==1) or (mf==0 and mb==2): return 9, '九等奖'
    return 0, ''
def check_single_7xc(u_nums, d_nums):
    hits = sum(1 for i in range(min(len(u_nums), len(d_nums))) if u_nums[i] == d_nums[i])
    if hits == 7: return 1, '特等奖'
    if hits == 6: return 2, '一等奖' 
    if hits >= 4: return 6, '五等奖' 
    return 0, ''
def calc_compound_win(ltype, bet_nums, draw, is_zj=False):
    d_red = draw['red'].split()
    d_blue = draw['blue'].split() if draw['blue'] else []
    if ltype == '7xc': d_blue = []
    u_reds, u_blues, combs_red, combs_blue = [], [], [], []
    try:
        if ltype == 'ssq':
            if '+' in bet_nums: p=bet_nums.split('+'); u_reds=p[0].replace(',',' ').split(); u_blues=p[1].replace(',',' ').split()
            else: raw=bet_nums.replace(',',' ').split(); u_reds=raw[:-1]; u_blues=[raw[-1]]
            combs_red, combs_blue = get_combinations(u_reds, 6), get_combinations(u_blues, 1)
        elif ltype == 'dlt':
            if '+' in bet_nums: p=bet_nums.split('+'); u_reds=p[0].replace(',',' ').split(); u_blues=p[1].replace(',',' ').split()
            else: raw=bet_nums.replace(',',' ').split(); u_reds=raw[:5]; u_blues=raw[5:]
            combs_red, combs_blue = get_combinations(u_reds, 5), get_combinations(u_blues, 2)
        elif ltype == '7xc':
            u_reds=bet_nums.replace(',',' ').split(); combs_red=[u_reds]; combs_blue=[[]]
    except: return {'is_win': False, 'details': {}}
    prize_map = {}
    try:
        for p in json.loads(draw['prizes']): prize_map[p['n']] = float(p['m'].replace(',',''))
    except: pass
    summary, total_money = {}, 0
    for r_c in combs_red:
        for b_c in combs_blue:
            tier, name = 0, ''
            if ltype == 'ssq': tier, name = check_single_ssq(r_c, b_c[0], d_red, d_blue[0])
            elif ltype == 'dlt': tier, name = check_single_dlt(r_c, b_c, d_red, d_blue)
            elif ltype == '7xc': tier, name = check_single_7xc(r_c, d_red)
            if tier > 0:
                money = prize_map.get(name, 0)
                if ltype == 'dlt' and is_zj and tier <= 2: money += prize_map.get(name+"(追加)", 0)
                if name not in summary: summary[name] = {'count': 0, 'money': 0}
                summary[name]['count'] += 1; summary[name]['money'] += money; total_money += money
    hit_red = list(set(u_reds) & set(d_red))
    hit_blue = list(set(u_blues) & set(d_blue))
    return {'total_money': total_money, 'details': summary, 'hit_red': hit_red, 'hit_blue': hit_blue, 'is_win': total_money > 0 or len(summary) > 0}

# --- 路由 ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/init')
def api_init():
    conn = get_db()
    resp = {}
    for t in URLS.keys():
        try:
            rows = conn.execute(f'SELECT issue, date, red, blue, prizes FROM history WHERE type="{t}" ORDER BY date DESC').fetchall()
            issues = conn.execute(f'SELECT issue, date FROM history WHERE type="{t}" ORDER BY date DESC').fetchall()
            resp[t] = {
                'history': [dict(r) for r in rows],
                'issues': [dict(r) for r in issues],
                'next_draw': get_next_event_time(t, 'draw_time'),
                'next_stop': get_next_event_time(t, 'stop_time')
            }
        except: 
            resp[t] = {'history':[], 'issues':[], 'next_draw': '...', 'next_stop': '...'}
    conn.close()
    return jsonify(resp)

# 调试接口：查看调度器状态
@app.route('/api/status')
def api_status():
    return jsonify({
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scheduler": SCHEDULER_STATUS,
        "timezone_check": time.tzname
    })

@app.route('/api/check', methods=['POST'])
def api_check():
    data = request.json
    ltype, bets, mode, issue = data.get('type'), data.get('bets'), data.get('mode'), data.get('issue')
    conn = get_db()
    try:
        sql = f"SELECT * FROM history WHERE type='{ltype}'"
        if mode == 'current' and issue != 'latest': sql += f" AND issue='{issue}'"
        sql += " ORDER BY date DESC"
        if mode == 'current' and issue == 'latest': sql += " LIMIT 1"
        draws = conn.execute(sql).fetchall()
    except: draws = []
    conn.close()
    results = []
    for bet in bets:
        b_str = bet['nums'].strip()
        if not b_str: continue
        matches = []
        for d in draws:
            res = calc_compound_win(ltype, b_str, d, data.get('zhuijia', False))
            if res['is_win']: matches.append({'issue': d['issue'], 'date': d['date'], 'draw_red': d['red'], 'draw_blue': d['blue'], 'win_data': res})
        if matches or mode == 'current': results.append({'bet': b_str, 'matches': matches})
    return jsonify(results)

# --- 关键修改：Gunicorn 启动入口 ---
# 这里的代码在 Gunicorn 加载 app 时就会执行
print(f"[{datetime.now()}] 正在启动服务进程...", flush=True)
init_db()
# 启动后台守护线程
bg_thread = Thread(target=smart_scheduler, daemon=True)
bg_thread.start()

if __name__ == '__main__':
    # 仅在本地 python app.py 运行时执行
    app.run(host='0.0.0.0', port=PORT)
