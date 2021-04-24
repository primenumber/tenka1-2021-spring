#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <queue>
#include <random>
#include <mutex>
#include <atomic>
using namespace std;

const int GAME_INFO_SLEEP_TIME = 5000;
const int UPDATE_CACHE_TIME = 20000;

mt19937 mt;

struct MasterData {
	int game_period;
	int max_len_task;
	int num_agent;
	vector<pair<int, int>> checkpoints;
	int area_size;
};

struct AgentMove {
	double x;
	double y;
	int t;
};

struct Agent {
	vector<AgentMove> move;
	string history;
};

struct Task {
	string s;
	int t;
	int weight;
	int count;
	int total;
};

struct Game {
	int now;
	vector<Agent> agent;
	vector<Task> task;
	int next_task;
};

struct Move {
	int now;
	vector<AgentMove> move;
};

MasterData call_master_data() {
	cout << "master_data" << endl;
	MasterData res;
	cin >> res.game_period >> res.max_len_task >> res.num_agent;
	res.checkpoints = vector<pair<int, int>>(26);
	for (auto& c : res.checkpoints) {
		cin >> c.first >> c.second;
	}
	cin >> res.area_size;
	return res;
}

Game call_game() {
	cout << "game" << endl;
	Game res;
	int num_agent, num_task;
	cin >> res.now >> num_agent >> num_task;
	res.agent.resize(num_agent);
	for (auto& a : res.agent) {
		int num_move;
		cin >> num_move;
		a.move.resize(num_move);
		for (auto& m : a.move) {
			cin >> m.x >> m.y >> m.t;
		}
		string h;
		cin >> h;
		a.history = h.substr(1);
	}
	res.task.resize(num_task);
	for (auto& t : res.task) {
		cin >> t.s >> t.t >> t.weight >> t.count >> t.total;
	}
	cin >> res.next_task;
	return res;
}

Move read_move() {
	Move res;
	int num_move;
	cin >> res.now >> num_move;
	res.move.resize(num_move);
	for (auto& m : res.move) {
		cin >> m.x >> m.y >> m.t;
	}
	return res;
}

Move call_move(int index, int x, int y) {
	cout << "move " << index << "-" << x << "-" << y << endl;
	return read_move();
}

Move call_move_next(int index, int x, int y) {
	cout << "move_next " << index << "-" << x << "-" << y << endl;
	return read_move();
}

int count_occur(const string& order, const string& task) {
  int result = 0;
  auto searcher = default_searcher(begin(task), end(task));
  auto first = begin(order);
  for (auto itr = search(first, end(order), searcher); itr != end(order); itr = search(first, end(order), searcher)) {
    ++result;
    first = next(itr);
  }
  return result;
}

double distance(const pair<int,int> from, const pair<int,int> to) {
  int dx = abs(from.first - to.first);
  int dy = abs(from.second - to.second);
  return sqrt(dx * dx + dy * dy);
}

double efficiency(const string& order, const vector<Task>& tasks, const MasterData& master_data) {
  double score = 0;
  for (auto&& task : tasks) {
    double occur = count_occur(order, task.s);
    score += occur * task.weight / (task.total + 1);
  }
  double cost = 0;
  int len = size(order);
  for (int i = 0; i < len-1; ++i) {
    int from = order[i] - 'A';
    int to = order[i+1] - 'A';
    if (from != to) {
      cost += distance(master_data.checkpoints[from], master_data.checkpoints[to]);
    } else {
      cost += 2;
    }
  }
  if (len == 1) cost += 2;
  return score / cost;
}

string concat_merge(const string& l, const string& r, size_t max_len) {
  size_t ll = size(l), rl = size(r);
  for (int len = size(r) - 1; len > 0; --len) {
    if (ll + rl - len > max_len) return "";
    if (equal(end(l) - len, end(l), begin(r), begin(r) + len)) {
      string tmp(size(l) + size(r) - len, 'A');
      copy(begin(l), end(l), begin(tmp));
      copy(begin(r) + len, end(r), begin(tmp) + size(l));
      return tmp;
    }
  }
  return l + r;
}

vector<string> generate_candidates(const vector<Task>& tasks) {
  unordered_set<string> candidates;
  queue<string> que;
  for (char front = 'A'; front <= 'Z'; ++front) {
    for (auto&& task : tasks) {
      if (task.weight <= 100) continue;
      string s(1, front);
      s += task.s;
      que.push(s);
      candidates.insert(s);
    }
  }
  while (!que.empty()) {
    auto s = que.front();
    que.pop();
    for (auto&& task : tasks) {
      if (task.weight <= 100) continue;
      auto t = concat_merge(s, task.s, 19);
      if (t == "") continue;
      if (size(t) > 19) continue;
      if (size(t) == size(s)) continue;
      if (candidates.count(t)) continue;
      candidates.insert(t);
      que.push(move(t));
    }
  }
  return vector<string>(begin(candidates), end(candidates));
}

struct GoodCache {
  vector<string> cache;
  mutex mtx;
  vector<string> candidates;
  GoodCache() : cache(26, "A") {}
  void update(const Game& game_info, const MasterData& master_data, bool update_candi = false) {
    std::thread t([&] (Game game_info, MasterData master_data) {
        using clock = chrono::high_resolution_clock;
        const auto start = clock::now();
        if (update_candi) {
          candidates = generate_candidates(game_info.task);
        }
        const auto middle = clock::now();
        int len = size(candidates);
        mutex mtx2;
        vector<double> current_eff(26, 0.0);
        vector<string> result(26);
#pragma omp parallel for
        for (int i = 0; i < len; ++i) {
          auto eff = efficiency(candidates[i], game_info.task, master_data);
          char c = candidates[i].front();
          int idx = c - 'A';
          lock_guard lock(mtx2);
          if (eff > current_eff[idx]) {
            current_eff[idx] = eff;
            result[idx] = candidates[i];
          }
        }
        const auto finish = clock::now();
        const chrono::duration<double, std::milli> mid_e = middle - start;
        const chrono::duration<double, std::milli> elapsed = finish - start;
        {
          lock_guard lock(mtx);
          cache = result;
        }
        cerr << "Updated cache: " << elapsed.count() << "ms, gen: " << mid_e.count() << "ms" << endl;
    }, game_info, master_data);
    if (!update_candi) {
      t.detach();
    } else {
      t.join();
    }
  }
  void update_old(const Game& game_info, const MasterData& master_data) {
    std::thread t([&] (Game game_info, MasterData master_data) {
        vector<string> result(26);
        using clock = chrono::high_resolution_clock;
        const auto start = clock::now();
#pragma omp parallel for
        for (char c = 'A'; c <= 'Z'; ++c) {
          string s(1, c);
          double current_eff = 0;
          string tmp;
          for (auto&& task : game_info.task) {
            if (task.weight <= 100) continue;
            auto t = concat_merge(s, task.s, 100);
            for (auto&& another_task : game_info.task) {
              if (another_task.weight <= 100) continue;
              auto u = concat_merge(t, another_task.s, 100);
              if (size(u) <= 15) {
                for (auto&& third_task : game_info.task) {
                  if (third_task.weight <= 100) continue;
                  auto v = concat_merge(u, third_task.s, 100);
                  if (size(v) > 25) continue;
                  auto eff = efficiency(v, game_info.task, master_data);
                  if (eff > current_eff) {
                    current_eff = eff;
                    tmp = v;
                  }
                }
              }
              auto eff = efficiency(u, game_info.task, master_data);
              if (eff > current_eff) {
                current_eff = eff;
                tmp = u;
              }
            }
            auto eff = efficiency(t, game_info.task, master_data);
            if (eff > current_eff) {
              current_eff = eff;
              tmp = t;
            }
          }
          result[c - 'A'] = tmp;
        }
        const auto finish = clock::now();
        const chrono::duration<double, std::milli> elapsed = finish - start;
        {
          lock_guard lock(mtx);
          cache = result;
        }
        cerr << "Updated cache: " << elapsed.count() << "ms" << endl;
    }, game_info, master_data);
    t.detach();
  }
};

struct Bot {
	MasterData master_data;
	Game game_info;
	chrono::system_clock::time_point start_time;
	int start_game_time_ms;
	int next_call_game_info_time_ms;
  int next_call_update_good_time_ms;
	vector<int> agent_move_finish_ms;
	vector<queue<pair<int, int>>> agent_move_point_queue;
	vector<pair<int, int>> agent_last_point;
  GoodCache good_cache;
	Bot() {
		master_data = call_master_data();
		game_info = call_game();
		start_game_time_ms = game_info.now;
		cerr << "Start:" << start_game_time_ms << endl;
		start_time = chrono::system_clock::now();
		next_call_game_info_time_ms = get_now_game_time_ms() + GAME_INFO_SLEEP_TIME;
		next_call_update_good_time_ms = get_now_game_time_ms() + UPDATE_CACHE_TIME;
		agent_move_finish_ms.resize(master_data.num_agent);
		agent_move_point_queue.resize(master_data.num_agent);
		agent_last_point.resize(master_data.num_agent);
    good_cache.update(game_info, master_data, true);
		for (int i = 0; i < master_data.num_agent; ++ i) {
			agent_last_point[i] = {(int)game_info.agent[i].move.back().x, (int)game_info.agent[i].move.back().y};
			set_move_point(i);
		}
	}
	string choice_task(int index) {
    int current_idx = 0;
    for (int i = 0; i < size(master_data.checkpoints); ++i) {
      if (master_data.checkpoints[i] == agent_last_point[index]) {
        current_idx = i;
        break;
      }
    }
    lock_guard lock(good_cache.mtx);
    return good_cache.cache.at(current_idx);
	}
	pair<int, int> get_checkpoint(char c) {
		return master_data.checkpoints[c - 'A'];
	}
	// 移動予定を設定
	void set_move_point(int index) {
		const auto next_task = choice_task(index);
		cerr << "Agent#" << index+1 << " next task:" << next_task << endl;

		for (char c : next_task) {
			auto before_point = agent_last_point[index];
			auto move_point = get_checkpoint(c);

			// 移動先が同じ場所の場合判定が入らないため別の箇所に移動してからにする
			if (before_point == move_point) {
        int d[] = {0, 1, 0, -1, 0};
        for (int i = 0; i < 4; ++i) {
          int tmp_x = before_point.first + d[i];
          int tmp_y = before_point.second + d[i+1];
          if (tmp_x >= 0 && tmp_x <= master_data.area_size && tmp_y >= 0 && tmp_y <= master_data.area_size) {
            agent_move_point_queue[index].push({tmp_x, tmp_y});
            break;
          }
        }
			}

			agent_move_point_queue[index].push(move_point);
			agent_last_point[index] = move_point;
		}
	}
	int get_now_game_time_ms() {
		auto t = chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - start_time).count();
		return start_game_time_ms + t;
	}
	Move move_next(int index) {
		auto [x, y] = agent_move_point_queue[index].front();
		agent_move_point_queue[index].pop();
		auto move_next_res = call_move_next(index+1, x, y);
		cerr << "Agent#" << index+1 << " move_next to (" << x << ", " << y << ")" << endl;

		agent_move_finish_ms[index] = move_next_res.move[1].t + 100;

		// タスクを全てやりきったら次のタスクを取得
		if (agent_move_point_queue[index].empty()) {
			set_move_point(index);
		}

		return move_next_res;
	}
	double get_now_score() {
		double score = 0.0;
		for (const auto& task : game_info.task) {
			if (task.total == 0) continue;
			score += (double)(task.weight * task.count) / task.total;
		}

		return score;
	}
	void solve() {
		for (;;) {
			int now_game_time_ms = get_now_game_time_ms();

			// エージェントを移動させる
      vector<pair<int,int>> vp;
			for (int i = 0; i < master_data.num_agent; ++ i) {
				if (agent_move_finish_ms[i] < now_game_time_ms) {
          vp.emplace_back(agent_move_finish_ms[i], i);
        }
      }
      sort(begin(vp), end(vp));
      for (auto&& [t, i] : vp) {
        auto move_next_res = move_next(i);
        // 次の移動予定がない場合もう一度実行する
        if (move_next_res.move.size() == 2) {
          move_next(i);
				}
			}

			// ゲーム情報更新
			if (next_call_game_info_time_ms < now_game_time_ms) {
				cerr << "Update GameInfo" << endl;
				game_info = call_game();
				next_call_game_info_time_ms = get_now_game_time_ms() + GAME_INFO_SLEEP_TIME;
				cerr << "Score: " << get_now_score() << endl;
				cerr << "Start Update GoodCache" << endl;
				good_cache.update(game_info, master_data, false);
			}

			//// キャッシュ更新
			//if (next_call_update_good_time_ms < now_game_time_ms) {
			//	next_call_update_good_time_ms = get_now_game_time_ms() + UPDATE_CACHE_TIME;
			//}

			this_thread::sleep_for(chrono::milliseconds(100));
		}
	}
};

int main() {
	random_device seed_gen;
	mt = mt19937(seed_gen());

	Bot bot;
	bot.solve();
}
