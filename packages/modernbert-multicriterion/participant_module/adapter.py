"""APCTT adapter with a fine-tuned ModernBERT evidence classifier."""
import logging
import os
from pathlib import Path

from .research.catalog import DEFAULT_MODELS, default_model_paths, resolve_catalog_model
from .research.criteria_policy import parse_criteria
from .research.pdf_parser import read_pdf, extract_windows, exclude_front_matter
from .research.two_stage_retrieval import TwoStageRetriever


MODERNBERT_ID = "modernbert-base"
MODERNBERT_PATH = "answerdotai/ModernBERT-base/8949b909ec900327062f0ebf497f51aef5e6f0c8"


class AssessmentModule:
    def load(self, config, asset_root):
        import torch
        from .research.modernbert_evidence import ModernBertEvidence

        self.checklist=config['checklist']
        settings=config.get('module',{}).get('settings',{}) or {}
        device=config.get('compute',{}).get('device','cpu')
        if device=='cpu':
            torch.set_num_threads(settings.get('cpu_threads',4))
        store=Path(os.environ['APCTT_MODEL_STORE'])
        catalogue=settings.get('catalog_path') or os.environ.get('APCTT_MODEL_CATALOG')
        defaults=default_model_paths(store)
        paths={role:resolve_catalog_model(DEFAULT_MODELS[role][0],catalogue,store) if catalogue else defaults[role]
               for role in ('encoder','reranker')}
        base = resolve_catalog_model(MODERNBERT_ID, catalogue, store) if catalogue else store / MODERNBERT_PATH
        self.retriever = TwoStageRetriever(str(paths['encoder']), str(paths['reranker']), device=device,
                                           candidate_k=20, final_k=4)
        checkpoint = Path(asset_root) / "modernbert_trl"
        self.judge = ModernBertEvidence(checkpoint, device, tokenizer_path=base)

    def assess(self,case,context):
        criteria=self.checklist['criteria']
        parsed = parse_criteria(self.checklist)
        levels = {row.id: row.trl for row in parsed}
        ids=[row['criterion_id'] for row in criteria]
        target=str(case['assessment_target'].get('description') or case['assessment_target']['target_id'])
        response=dict(schema_version=case['schema_version'],checklist_version=self.checklist['version'],
                      case_id=case['case_id'],assessment_target=case['assessment_target'],
                      assessment_status='abstained',predicted_trl=None,abstention_reason='insufficient evidence',
                      error_code=None,criterion_status=dict.fromkeys(ids,'insufficient evidence'),
                      evidence_reference={key:[] for key in ids},missing_evidence={})
        windows = []
        for doc in case["documents"]:
            pages = read_pdf(doc["path"], doc["document_id"], remaining_ms=context.remaining_ms)
            windows.extend(extract_windows(exclude_front_matter(pages)))

        texts = [item.text for item in windows]
        stage_criteria = [criterion for criterion in parsed if criterion.trl]
        quality_criteria = [criterion for criterion in parsed if not criterion.trl]
        hits_by_criterion = {
            criterion.id: self.retriever.retrieve(f"{target}. {criterion.hypothesis}", texts)
            for criterion in parsed
        }
        pool = sorted({index for criterion in stage_criteria
                       for index, _ in hits_by_criterion[criterion.id]})
        stage_candidates = [windows[index] for index in pool]
        evidence = {}
        for criterion in stage_criteria + quality_criteria:
            candidates = (stage_candidates if criterion.trl else
                          [windows[index] for index, _ in hits_by_criterion[criterion.id]])
            scores = self.judge.support(target, criterion.hypothesis, candidates, context.remaining_ms, 1500)
            if scores is None:
                response.update(assessment_status="failed", predicted_trl=None,
                                abstention_reason=None, error_code="ASSESSMENT_TIME_BUDGET")
                return response
            if not scores:
                continue
            score, item = max(zip(scores, candidates), key=lambda pair: pair[0])
            threshold = 0.98 if criterion.trl >= 8 else 0.75 if criterion.trl else 0.50
            if score >= threshold:
                evidence[criterion.id] = (score, item)
        for key, (score, item) in evidence.items():
            response["criterion_status"][key] = "met"
            response["evidence_reference"][key] = [
                dict(document_id=item.source_id, page=item.page, supporting_span=item.text)
            ]
            logging.getLogger(__name__).info("%s score=%.4f page=%s", key, score, item.page)
        supported=[level for key,level in levels.items() if level and response['criterion_status'][key]=='met']
        if supported:
            response.update(assessment_status='completed',predicted_trl=max(supported),abstention_reason=None,
                            trl_rationale='Highest rubric level supported by a verified excerpt.')
        response['missing_evidence']={key:'No accepted supporting excerpt.' for key in ids
                                      if response['criterion_status'][key]!='met'}
        return response

    def reset_case(self):
        pass

    def close(self):
        self.judge=None
        self.retriever=None
