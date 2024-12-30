# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 19:55
# @Author  : Xuebing Wang
# @Email   : wangxuebing0212@163.com
# @FileName: utils.py
# @Description: there is code description.

def model_down(model_name,save_path='./'):
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(model_id=model_name, cache_dir=save_path)

if __name__ == '__main__':
    print('保持好心情！ ')
    model_name = 'tiansz/bert-base-chinese'
    save_path = 'pretrain_models/'
    # model_down(model_name,save_path)
