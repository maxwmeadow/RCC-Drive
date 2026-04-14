RCC-Drive Trainval Subset
========================================

Total samples: 118
Locations: {'boston-seaport': 75, 'singapore-onenorth': 37, 'singapore-queenstown': 4, 'singapore-hollandvillage': 2}
Relations: {'NTPP': 61, 'PO': 57}

Structure:
  images/           <- CAM_FRONT images, named by filename
  spatial_facts/    <- RCC-8 JSON per sample_token
  sample_tokens.json <- ordered list of tokens

For Phase 2 notebook:
  Replace KEPT_TOKENS with the list in sample_tokens.json
  Point SPATIAL_FACTS_DIR to the spatial_facts/ folder
  Point CAM_FRONT_DIR to the images/ folder
