
import yaml
import logging
from .generator import MinimalWeightGenerator
from ..config.manager import GenerationConfig

logger = logging.getLogger(__name__)

class BatchGenerator:
    """Batch generate models from a configuration file"""
    
    def __init__(self, config_file: str):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
            
    def generate_all(self):
        """Execute batch generation"""
        models = self.config.get('models', [])
        logger.info(f"Starting batch generation for {len(models)} models")
        
        for model_spec in models:
            model_id = model_spec['id']
            output_dir = model_spec['output']
            options = model_spec.get('options', {})
            
            logger.info(f"Batch processing: {model_id} -> {output_dir}")
            
            try:
                # Map options to GenerationConfig
                config = GenerationConfig(
                    max_shard_size=options.get('max_shard_size', "5GB"),
                    strategy=options.get('strategy', "random"),
                    dtype=options.get('dtype', "bfloat16")
                )
                
                generator = MinimalWeightGenerator(
                    model_id=model_id,
                    output_dir=output_dir,
                    config=config
                )
                generator.generate()
                logger.info(f"Successfully generated {model_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate {model_id}: {e}")
                # Continue to next model in batch
