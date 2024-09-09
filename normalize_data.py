import json
import chess

RAW_DATA_FILE = './dataset/lichess_db_eval.jsonl'

lines = []
with open(RAW_DATA_FILE, 'r') as f:
    for i in range(1_000_000):
        lines.append(f.readline())

with open('dataset/raw_1_000_000.jsonl', 'w') as f:
    f.writelines(lines)

normalized_lines = [] # {"fen": fen, "eval": eval}
for line in lines:
    json_line = json.loads(line)
    normalized_line = {"fen": json_line["fen"], "eval": None}
    average_eval = 0.0
    mate_flag = False
    for eval in json_line['evals']:
        for pv in eval['pvs']:
            if 'mate' in pv:
                mate_flag = True
            else:
                average_eval += pv['cp']
    average_eval /= len(json_line['evals'])
    normalized_line['eval'] = average_eval
    if not mate_flag:
        normalized_lines.append(normalized_line)

with open('dataset/normalized_1_000_000.jsonl', 'w') as f:
    for line in normalized_lines:
        f.write(json.dumps(line) + '\n')

# integer representations
piece_char_to_int = {
    '.': 0,

    # black
    'p': -1,
    'n': -2,
    'b': -3,
    'r': -4,
    'q': -5,
    'k': -6,

    # white
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
}

for line in normalized_lines:
    board = chess.Board(line['fen'])
    board_str = str(board)
    vectors = []
    for board_str_line in board_str.split('\n'):
        vector = [piece_char_to_int[c] for c in board_str_line.replace(' ', '').replace('\n', '')]
        vectors.extend(vector)
    line['vectors'] = vectors
    del line['fen']

with open('dataset/vectorized_1_000_000.jsonl', 'w') as f:
    for line in normalized_lines:
        f.write(json.dumps(line) + '\n')