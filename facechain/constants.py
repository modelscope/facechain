neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'
pos_prompt_with_cloth = 'raw photo, masterpiece, chinese, {}, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, slim body, photorealistic, best quality'
pos_prompt_with_style = '{}, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, slim body, photorealistic, best quality'

base_models = [
    {'name': 'leosamsMoonfilm_filmGrain20',
    'model_id': 'ly261666/cv_portrait_model',
    'revision': 'v2.0',
    'sub_path': "film/film",
    'style_list': ['默认风格(default style)', '凤冠霞帔(Chinese traditional gorgeous suit)']},
    {'name': 'MajicmixRealistic_v6',
    'model_id': 'YorickHe/majicmixRealistic_v6',
    'revision': 'v1.0.0',
    'sub_path': "realistic",
    'style_list': ['生日气球风(birthday balloon)', '冬季汉服(Chinese winter hanfu)', '校服风(School uniform)', '婚纱风(Wedding dress)', '拍立得风(Polaroid style)', '仙女风(Fairy style)', '古风(traditional Chinese Style)', '壮族服装风(zhuangzu)', '欧式田野风(european fields)']},
]

cloth_prompt = [
    {'name': '工作服(working suit)', 'prompt': 'wearing high-class business/working suit, simple background, high-class pure color background'},  # male and female
    {'name': '盔甲风(armor)', 'prompt': 'wearing silver armor, simple background, high-class pure color background'},  # male
    {'name': 'T恤衫(T-shirt)', 'prompt': 'wearing T-shirt, simple background, high-class pure color background'},  # male and female
    {'name': '汉服风(hanfu)', 'prompt': 'wearing beautiful traditional hanfu, upper_body, simple background, high-class pure color background'},  # female
    {'name': '女士晚礼服(gown)', 'prompt': 'wearing an elegant evening gown, simple background, high-class pure color background'},  # female
    {'name': '赛博朋克(cybernetics punk)', 'prompt': 'white hair, neon glowing glasses, cybernetics, punks, robotic, AI, NFT art, Fluorescent color, ustration'}, # male and female
]

styles = [
    {'name': '默认风格(default style)'},
    {'name': '凤冠霞帔(Chinese traditional gorgeous suit)',
     'model_id': 'ly261666/civitai_xiapei_lora',
     'revision': 'v1.0.0',
     'bin_file': 'xiapei.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '汉服风(hanfu)',
     'add_prompt_style': 'red, hanfu, tiara, crown'},
    {'name': '生日气球风(birthday balloon)',
     'model_id': '/mnt/workspace/facechain/facechain/balloon_loral' ,
     'revision': 'v1.0.0',
     'bin_file': 'balloon.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '生日气球风(birthday balloon)',
     'add_prompt_style': 'Balloon, Best quality, 1girl, <lora:Balloon_v01:0.8>'},
     {'name': '冬季汉服(Chinese winter hanfu)',
     'model_id': 'YorickHe/Winter_hanfu_lora',
     'revision': 'v1.0.0',
     'bin_file': 'Winter_Hanfu.safetensors',
     'multiplier_style': 0.3,
     'multiplier_human': 0.95,
     'cloth_name': '冬季汉服(winter hanfu)',
     'add_prompt_style': 'red hanfu, winter hanfu, cloak, photography, warm light, sunlight, majestic snow scene,  close-up, front view, soft falling snowflakes, jewelry, enchanted winter wonderland'},
     {'name': '校服风(School uniform)',
     'model_id': 'YorickHe/JK_uniform_lora',
     'revision': 'v1.0.0',
     'bin_file': 'jk_uniform.safetensors',
     'multiplier_style': 0.2,
     'multiplier_human': 0.95,
     'cloth_name': '校服风(school uniform)',
     'add_prompt_style': 'JK_style, white short-sleeved JK_shirt, dark blue JK_skirt, bow JK_tie,  close-up, looking at viewer, night, Tokyo street, night city scape'},
     {'name': '婚纱风(Wedding dress)',
     'model_id': 'YorickHe/outdoor_photo_lora',
     'revision': 'v1.0.0',
     'bin_file': 'outdoor_photo_v2.0.safetensors',
     'multiplier_style': 0.3,
     'multiplier_human': 0.95,
     'cloth_name': '户外婚纱照(Outdoor wedding photo)',
     'add_prompt_style': 'white wedding dress, 1girl, summer, flower, happy atmosphere, sunlight, Waist Shot,'},
     {'name': '拍立得风(Polaroid style)',
     'model_id': 'YorickHe/polaroid_lora',
     'revision': 'v1.0.0',
     'bin_file': 'InstantPhotoX3.safetensors',
     'multiplier_style': 0.35,
     'multiplier_human': 0.95,
     'cloth_name': '拍立得风(Polaroid style)',
     'add_prompt_style': '1girl, close-up, face shot, stylish outfit, fitted jeans, oversized jacket, fashionable accessories, cityscape backdrop, rooftop or high-rise balcony, dynamic composition, engaging pose, soft yet striking lighting, shallow depth of field, bokeh from city lights, naturally blurred background'},
     {'name': '仙女风(Fairy style)',
     'model_id': 'YorickHe/fairy_lora',
     'revision': 'v1.0.0',
     'bin_file': 'fairy.safetensors',
     'multiplier_style': 0.25,
     'multiplier_human': 0.95,
     'cloth_name': '仙女风(Fairy style)',
     'add_prompt_style': 'a beautiful fairy standing in the middle of a flower field, petals, close-up, warm light, light green atmosphere, white atmosphere, in the style of celebrity photography, soft, romantic scenes, flowing fabrics, light white and light orange, high resolution'},
     {'name': '古风(traditional Chinese Style)',
     'model_id': 'iotang/lora_testing',
     'revision': 'v5',
     'bin_file': 'MoXinV1.safetensors',
     'multiplier_style': 0.3,
     'multiplier_human': 0.95,
     'cloth_name': '汉服风(hanfu)',
     'add_prompt_style': '(ultra high res face, face ultra zoom, highres, best quality, ultra detailed, cinematic lighting, portrait, Chinese traditional ink painting:1.2), sfw, shuimobysim, song, anxiang, hanfu, Ultra HD, wuchangshuo, detailed background, looking at viewer, serenity, peace'},
    {'name': '壮族服装风(zhuangzu)',
     'model_id': 'iotang/lora_testing',
     'revision': 'v5',
     'bin_file': 'zhuangnv.safetensors',
     'multiplier_style': 0.7,
     'multiplier_human': 0.95,
     'cloth_name': '壮族服装风(zhuangzu)',
     'add_prompt_style': '(masterpiece, ultra high res face, face ultra zoom, highres, best quality, ultra detailed, cinematic lighting, portrait:1.2), sfw, facing the camera with a smile, zhuangzunv, ornaments, jewelry, headwear, beautiful embroidery, floral print, marvelous design, ancient Chinese traditional clothing'},
    {'name': '欧式田野风(european fields)',
     'model_id': 'iotang/lora_testing',
     'revision': 'v5',
     'bin_file': 'edgEuropean_Vintage.safetensors',
     'multiplier_style': 0.55,
     'multiplier_human': 0.95,
     'cloth_name': '欧式田野风(european fields)',
     'add_prompt_style': '(masterpiece, ultra high res face, face ultra zoom, highres, best quality, ultra detailed, detailed background, cinematic lighting, portrait:1.2), sfw, focused, edgEV, wearing edgEV_vintage dress, field, natural lighting, windy hair, gentle hair, clean'},
]

pose_models = [
    {'name': '无姿态控制(No pose control)'},
    {'name': 'pose-v1.1-with-depth'},
    {'name': 'pose-v1.1'}
]

pose_examples = {
    'man': [
        ['./poses/man/pose1.png'],
        ['./poses/man/pose2.png'],
        ['./poses/man/pose3.png'],
        ['./poses/man/pose4.png']
    ],
    'woman': [
        ['./poses/woman/pose1.png'],
        ['./poses/woman/pose2.png'],
        ['./poses/woman/pose3.png'],
        ['./poses/woman/pose4.png'],
    ]
}
