import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).parent
REPO = Path('/home/tomasruizt/code/vllm')
PYTHON = '/home/tomasruizt/.venv/bin/python'
CACHE = REPO / '.cache/huggingface/hub'
PORT = 8023
COUNTERS = (
    'vllm:spec_decode_num_drafts_total',
    'vllm:spec_decode_num_draft_tokens_total',
    'vllm:spec_decode_num_accepted_tokens_total',
)


def main():
    target = 'RedHatAI/GLM-5.3-NVFP4'
    draft = 'RedHatAI/GLM-5.3-speculator.dspark'
    revision = lambda model: (CACHE / ('models--' + model.replace('/', '--')) / 'refs/main').read_text().strip()
    spec = {
        'method': 'dspark', 'model': draft, 'revision': revision(draft),
        'num_speculative_tokens': 8, 'attention_backend': 'FLASH_ATTN',
        'draft_sample_method': 'probabilistic', 'enable_adaptive_verification': True,
    }
    command = [
        '/home/tomasruizt/.venv/bin/vllm', 'serve', target,
        '--revision', revision(target), '--served-model-name', 'glm-5.3',
        '--tensor-parallel-size', '1', '--data-parallel-size', '4',
        '--enable-expert-parallel', '--all2all-backend', 'allgather_reducescatter',
        '--attention-backend', 'FLASHINFER_MLA_SPARSE',
        '--kv-cache-dtype', 'fp8_e4m3', '--block-size', '64',
        '--max-model-len', '16384', '--max-num-seqs', '128',
        '--max-num-batched-tokens', '16384', '--gpu-memory-utilization', '0.85',
        '--reasoning-parser', 'glm45', '--chat-template-content-format', 'string',
        '--trust-remote-code', '--disable-uvicorn-access-log',
        '--host', '127.0.0.1', '--port', str(PORT),
        '--compilation-config', json.dumps({
            'cudagraph_mode': 'FULL_AND_PIECEWISE', 'max_cudagraph_capture_size': 1152,
        }),
        '--speculative-config', json.dumps(spec),
    ]
    env = os.environ | {
        'PATH': '/home/tomasruizt/.venv/bin:' + os.environ['PATH'],
        'HF_HUB_CACHE': str(CACHE), 'VLLM_USE_V2_MODEL_RUNNER': '1',
        'VLLM_ENGINE_READY_TIMEOUT_S': '3600',
    }
    result = {
        'command': command,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
        'target_revision': revision(target), 'draft_revision': revision(draft),
        'runs': [], 'status': 'starting',
    }
    save(result)
    print(json.dumps(result), flush=True)
    with (ROOT / 'server.log').open('w') as log:
        server = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / 'server.pid').write_text(str(server.pid))
        try:
            deadline = time.monotonic() + 3600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited with code {server.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server startup exceeded one hour')
                    time.sleep(5)
            print('Server healthy; warming up', flush=True)
            evaluate('warmup', 32, 64, 512, env)
            time.sleep(10)
            for concurrency in (8, 64):
                label = f'gsm8k-c{concurrency}'
                before = metrics(f'{label}-metrics-before.txt')
                print(f'Running full GSM8K at concurrency {concurrency}', flush=True)
                run = evaluate(label, concurrency, 1319, 2048, env)
                time.sleep(10)
                after = metrics(f'{label}-metrics-after.txt')
                delta = {key: after[key] - before[key] for key in COUNTERS}
                drafts, drafted, accepted = (delta[key] for key in COUNTERS)
                if drafts <= 0 or drafted <= 0:
                    raise RuntimeError(f'Missing speculative decoding activity: {delta}')
                run.update(concurrency=concurrency, mean_acceptance_length=1 + accepted / drafts,
                           draft_acceptance_rate=accepted / drafted, counter_deltas=delta)
                (ROOT / f'{label}.json').write_text(json.dumps(run, indent=2) + '\n')
                result['runs'].append(run)
                result['status'] = 'running'
                save(result)
                print(json.dumps(run), flush=True)
            result['status'] = 'completed'
        except Exception as exc:
            result.update(status='failed', error=str(exc))
            raise
        finally:
            save(result)
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def evaluate(label, concurrency, questions, max_tokens, env):
    output = ROOT / f'{label}.json'
    log_path = ROOT / f'{label}-client.log'
    with log_path.open('w') as log:
        subprocess.run([
            PYTHON, 'tests/evals/gsm8k/gsm8k_eval.py', '--port', str(PORT),
            '--num-questions', str(questions), '--num-shots', '5',
            '--max-tokens', str(max_tokens), '--temperature', '0', '--seed', '42',
            '--max-concurrency', str(concurrency), '--request-timeout-seconds', '1800',
            '--save-results', str(output),
        ], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    if 'Error calling vLLM' in log_path.read_text():
        raise RuntimeError(f'Failed API requests in {log_path}')
    return json.loads(output.read_text())


def metrics(filename):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/metrics', timeout=30) as response:
        text = response.read().decode()
    (ROOT / filename).write_text(text)
    values = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        match = re.match(r'(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total)(?:\{[^}]*\})?\s+(\S+)', line)
        if match:
            key, value = match.groups()
            values[key] += float(value)
    return values


def save(result):
    (ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
