neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'
pos_prompt_with_cloth = 'raw photo, masterpiece, chinese, {}, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, slim body, photorealistic, best quality'
pos_prompt_with_style = '{}, upper_body, raw photo, masterpiece, chinese, solo, medium shot, high detail face, slim body, photorealistic, best quality'


# cloth_prompt = [
#     {'name': '工作服(working suit)', 'prompt': 'high-class business/working suit'},  # male and female
#     {'name': '盔甲风(armor)', 'prompt': 'silver armor'},  # male
#     {'name': 'T恤衫(T-shirt)', 'prompt': 'T-shirt'},  # male and female
#     {'name': '汉服风(hanfu)', 'prompt': 'beautiful traditional hanfu, upper_body'},  # female
#     {'name': '女士晚礼服(gown)', 'prompt': 'an elegant evening gown'},  # female
# ]
cloth_prompt = [
    {'name': '工作服(working suit)', 'prompt': 'wearing high-class business/working suit, simple background, high-class pure color background'},  # male and female
    {'name': '盔甲风(armor)', 'prompt': 'wearing silver armor, simple background, high-class pure color background'},  # male
    {'name': 'T恤衫(T-shirt)', 'prompt': 'wearing T-shirt, simple background, high-class pure color background'},  # male and female
    {'name': '汉服风(hanfu)', 'prompt': 'wearing beautiful traditional hanfu, upper_body, simple background, high-class pure color background'},  # female
    {'name': '女士晚礼服(gown)', 'prompt': 'wearing an elegant evening gown, simple background, high-class pure color background'},  # female
    {'name': '赛博朋克(cybernetics punk)', 'prompt': 'white hair, futuristic glasses, neon glowing glasses, neon glowing eyes,  (Glowing neon:1.4), cybernetics, punks, robotic arm, AI, NFT art, Fluorescent color, ustration'}, # male and female
]


styles = [
    {'name': '默认风格(default style)'},
    {'name': '凤冠霞帔(Chinese traditional gorgeous suit)',
     'model_id': 'ly261666/civitai_xiapei_lora',
     'revision': 'v1.0.0',
     'bin_file': 'xiapei.safetensors',
     'multiplier_style': 0.35,
     'cloth_name': '汉服风(hanfu)',
     'add_prompt_style': 'red, hanfu, tiara, crown, '},
]


