import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from recursive_flux_engine import RecursiveFluxEngine

class FluxStreamingController:
    """
    Implements 'The Matios Fix': Parallelizing Ingestion and Thinking.
    """
    def __init__(self, model: RecursiveFluxEngine):
        self.model = model
        self.states = {}

    @torch.inference_mode()
    def process_stream(self, chunk_generator):
        prev_ingested = None
        prev_trigger = None

        for chunk in chunk_generator:
            # 1. Ingestion of current chunk (N+1)
            x_emb = self.model.embedding(chunk)
            # Handle tuple returns from modules
            x_ing, s_ing = self.model.ingestion(x_emb, self.states.get('ing'))
            ltc_res, trigger_prob, s_det = self.model.detector(x_ing, self.states.get('det'))

            self.states['ing'] = s_ing
            self.states['det'] = s_det

            # 2. Thinking about the PREVIOUS chunk (N)
            if prev_ingested is not None:
                x_thought, _, s_think = self.model.thinker(prev_ingested, prev_trigger, self.states.get('think'))
                self.states['think'] = s_think

                # Output Stage
                out, s_dec = self.model.decoder(x_thought, self.states.get('dec'))
                self.states['dec'] = s_dec

                yield self.model.output_head(self.model.final_norm(out))

            prev_ingested = x_ing + ltc_res
            prev_trigger = trigger_prob

        # Final cleanup pass
        x_thought, _, s_think = self.model.thinker(prev_ingested, prev_trigger, self.states.get('think'))
        out, s_dec = self.model.decoder(x_thought, self.states.get('dec'))
        yield self.model.output_head(self.model.final_norm(out))
