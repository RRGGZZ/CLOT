import torch
from torch import nn
import os
import copy

def export_policy_as_jit(actor_critic, path, exported_policy_name):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict, use_transformer):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        # import ipdb;ipdb.set_trace()
        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        if hasattr(actor, 'module'):
            # 如果是DDP模型，提取内部模块
            actor = actor.module
        elif hasattr(actor, '_forward_module'):
            # 如果是Lightning Fabric包装的模型
            actor = actor._forward_module
        else:
            actor = actor

        # 确保模型在eval模式
        actor.eval()

        class PPOWrapper(nn.Module):
            def __init__(self, actor, use_transformer):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOWrapper, self).__init__()
                self.actor = actor
            if use_transformer:
                def forward(self, actor_obs_current, actor_obs_past, actor_obs_future):
                    """
                    Dynamically creates a dictionary from the input keys and args.
                    """
                    return self.actor.act_inference_transformer(actor_obs_current, actor_obs_past, actor_obs_future)
            else:
                def forward(self, actor_obs):
                    """
                    Dynamically creates a dictionary from the input keys and args.
                    """
                    return self.actor.act_inference(actor_obs)

        wrapper = PPOWrapper(actor,use_transformer=use_transformer)
        if use_transformer:
            example_input_list = [example_obs_dict["actor_obs_current"], example_obs_dict["actor_obs_past"], example_obs_dict["actor_obs_future"]]
        else:
            example_input_list = example_obs_dict["actor_obs"]
            # print(1)
        
        # actor.double()
        # wrapper.double()
        # example_input_list.double()
        # import ipdb;ipdb.set_trace()
        # print('use_transformer:', use_transformer)
        if use_transformer:
            torch.onnx.export(
            wrapper,
            (example_input_list[0], example_input_list[1], example_input_list[2]), 
            path,
            verbose=True,
            input_names=["actor_obs_current","actor_obs_past","actor_obs_future"],  # Specify the input names
            output_names=["action"],       # Name the output
            # opset_version=17          # Specify the opset version, if needed
            export_params=True,
            opset_version=18,  # 提升到 17 以支持更多操作符
            do_constant_folding=True,
            )
        else:
            # print('hello')
            torch.onnx.export(
                wrapper,
                tuple(example_input_list),  # Pass x1 and x2 as separate inputs
                path,
                verbose=True,
                input_names=["actor_obs"],  # Specify the input names
                output_names=["action"],       # Name the output
                # opset_version=17          # Specify the opset version, if needed
                export_params=True,
                opset_version=18,  # 提升到 17 以支持更多操作符，避免 CastLike 等新操作符的兼容性问题
                do_constant_folding=True,
            )

def export_policy_and_estimator_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        left_hand_force_estimator = copy.deepcopy(inference_model['left_hand_force_estimator']).to('cpu')
        right_hand_force_estimator = copy.deepcopy(inference_model['right_hand_force_estimator']).to('cpu')

        class PPOForceEstimatorWrapper(nn.Module):
            def __init__(self, actor, left_hand_force_estimator, right_hand_force_estimator):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOForceEstimatorWrapper, self).__init__()
                self.actor = actor
                self.left_hand_force_estimator = left_hand_force_estimator
                self.right_hand_force_estimator = right_hand_force_estimator

            def forward(self, inputs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                actor_obs, history_for_estimator = inputs
                left_hand_force_estimator_output = self.left_hand_force_estimator(history_for_estimator)
                right_hand_force_estimator_output = self.right_hand_force_estimator(history_for_estimator)
                input_for_actor = torch.cat([actor_obs, left_hand_force_estimator_output, right_hand_force_estimator_output], dim=-1)
                return self.actor.act_inference(input_for_actor), left_hand_force_estimator_output, right_hand_force_estimator_output

        wrapper = PPOForceEstimatorWrapper(actor, left_hand_force_estimator, right_hand_force_estimator)
        example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["long_history_for_estimator"]]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs", "long_history_for_estimator"],  # Specify the input names
            output_names=["action", "left_hand_force_estimator_output", "right_hand_force_estimator_output"],       # Name the output
            opset_version=17           # 提升到 17 以支持更多操作符
        )