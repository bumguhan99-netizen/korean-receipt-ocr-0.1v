"""GLM-OCR model-only evaluation on the exact saved PaddleOCR Val sample list."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
from common import DEFAULT_DATA, read_csv, sha256, write_csv, write_json
from metrics import aggregate, measure

PROMPT = 'Text Recognition:'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/official_manifest.csv'))
    parser.add_argument('--model-dir', type=Path, default=Path('models/GLM-OCR'))
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--run-tag')
    args = parser.parse_args()
    if args.max_samples is not None and not 1 <= args.max_samples <= 100:
        parser.error('Smoke samples must be 1..100; omit for full Val')
    assert args.max_new_tokens > 0
    reference = Path('outputs/baseline/official/val') / ('smoke_100_seed42' if args.max_samples else 'full')
    reference_info = json.loads((reference/'run_info.json').read_text(encoding='utf-8'))
    assert reference_info['status'] == 'completed'
    assert sha256(args.manifest) == reference_info['manifest_sha256'], 'Manifest differs from Paddle baseline'
    rows = read_csv(reference/'samples.csv')
    if args.max_samples:
        rows = rows[:args.max_samples]
    assert rows and all(r['eval_split']=='val' for r in rows)
    assert len({r['image_path'] for r in rows}) == len(rows)
    tag = args.run_tag or (f'smoke_{args.max_samples}_seed42' if args.max_samples else 'full')
    out = Path('outputs/glm/official/val') / tag
    out.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', model='zai-org/GLM-OCR', started=datetime.now(timezone.utc).isoformat(),
                argv=sys.argv, python=sys.version, packages={k:importlib.metadata.version(k) for k in
                    ('torch','torchvision','transformers','huggingface-hub','Pillow')},
                manifest_name=args.manifest.name, manifest_sha256=sha256(args.manifest),
                reference_run=str(reference), reference_samples_sha256=sha256(reference/'samples.csv'),
                eval_split='val', sample_count=len(rows), seed=42, limited_to_n=args.max_samples,
                settings=dict(device=args.device, dtype='bfloat16' if args.device.startswith('cuda') else 'float32',
                    batch_size=1, cpu_threads=4, prompt=PROMPT, do_sample=False, num_beams=1,
                    max_new_tokens=args.max_new_tokens, attention='sdpa', normalization='Unicode NFC only',
                    tokenization='str.split()', postprocessing='decode generated suffix, skip special tokens, no stripping or text cleanup'),
                comparison_caveat='Same crop bytes/GT/metrics; model-native preprocessing/decoding and device differ from Paddle CPU runs')
    write_json(out/'run_info.json',info)
    predictions = []
    try:
        metadata = json.loads((args.model_dir/'download_manifest.json').read_text(encoding='utf-8'))
        assert metadata['repository'] == 'zai-org/GLM-OCR'
        for name,expected in metadata['files'].items():
            assert sha256(args.model_dir/name) == expected, f'Changed model asset: {name}'
        info['model_download'] = metadata
        for row in rows:
            path = (args.data_root/row['image_path']).resolve()
            assert path.is_relative_to(args.data_root.resolve())
            assert sha256(path)==row['image_sha256'], f'Changed image: {path}'
            with path.with_suffix('.txt').open(encoding='utf-8',newline='') as stream:
                assert stream.read().rstrip('\r\n')==row['gt_text'], f'Changed GT: {path}'
        write_csv(out/'samples.csv',rows)
        import torch
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration
        torch.set_num_threads(4)
        torch.manual_seed(42)
        if args.device.startswith('cuda'):
            assert torch.cuda.is_available(), 'CUDA unavailable; never silently fall back'
            info['gpu'] = torch.cuda.get_device_name(0)
            assert torch.cuda.is_bf16_supported(), 'BF16 unsupported'
        dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
        model = GlmOcrForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
                    dtype=dtype, attn_implementation='sdpa', trust_remote_code=False).to(args.device).eval()
        info['generation_config'] = model.generation_config.to_dict()
        info['processor_config'] = processor.to_dict() if hasattr(processor,'to_dict') else str(type(processor))
        info['model_load_seconds'] = time.perf_counter()-started
        write_json(out/'run_info.json',info)
        inferred = time.perf_counter()
        for index,row in enumerate(rows,1):
            before = time.perf_counter()
            messages = [{'role':'user','content':[
                {'type':'image','url':str((args.data_root/row['image_path']).resolve())},
                {'type':'text','text':PROMPT}]}]
            inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                         return_dict=True,return_tensors='pt').to(args.device)
            with torch.inference_mode():
                output = model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1)
            # Decoder-only generation returns the input prompt prefix as well.
            tokens = output[0,inputs['input_ids'].shape[1]:]
            prediction = processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert isinstance(prediction,str)
            record = dict(model='GLM-OCR',**{k:v for k,v in row.items() if k!='gt_text'},ground_truth=row['gt_text'],
                prediction=prediction,confidence=None,generated_tokens=len(tokens),
                token_limit_reached=int(len(tokens)>=args.max_new_tokens),seconds=time.perf_counter()-before,
                **measure(row['gt_text'],prediction))
            predictions.append(record)
            # Save after every crop for diagnosis; partial outputs are never reported as complete.
            with (out/'predictions.partial.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            if index==1:
                write_json(out/'first_result.json',dict(prediction=prediction,generated_token_ids=tokens.tolist(),
                           input_tokens=inputs['input_ids'].shape[1],image_path=row['image_path']))
            if index % 25 == 0 or index==len(rows):
                elapsed=time.perf_counter()-inferred
                print(f'GLM-OCR {args.device}: {index}/{len(rows)}, {elapsed:.1f}s, ETA {elapsed/index*(len(rows)-index):.0f}s',flush=True)
                info['completed_samples']=index
                write_json(out/'run_info.json',info)
            del inputs,output,tokens
        write_csv(out/'predictions.csv',predictions)
        metrics=aggregate(predictions)
        write_json(out/'metrics.json',metrics)
        fields=defaultdict(list)
        for row in predictions:
            fields[row['field']].append(row)
        write_csv(out/'by_field.csv',[dict(field=k,**aggregate(v)) for k,v in sorted(fields.items())])
        write_csv(out/'worst_50.csv',sorted(predictions,key=lambda r:(r['char_edits']/max(1,r['gt_chars']),r['char_edits']),reverse=True)[:50])
        write_csv(out/'summary.csv',[dict(model='GLM-OCR',**metrics,eval_split='val',manifest_name=args.manifest.name,
                limited_to_n=args.max_samples,device=args.device,normalization='NFC',tokenization='whitespace')])
        info.update(status='completed',completed_samples=len(predictions),finished=datetime.now(timezone.utc).isoformat(),
            inference_seconds=time.perf_counter()-inferred,token_limit_reached=sum(r['token_limit_reached'] for r in predictions),
            confidence_note='Not supplied: GLM generation is not PaddleOCR calibrated recognition confidence')
        write_json(out/'run_info.json',info)
        print(json.dumps(metrics,indent=2),flush=True)
    except BaseException:
        info.update(status='incomplete',completed_samples=len(predictions),error=traceback.format_exc())
        write_json(out/'run_info.json',info)
        (out/'error.log').write_text(info['error'],encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
