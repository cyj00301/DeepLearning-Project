from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, \
    recall_score, accuracy_score
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind



class Evaluation:
    @staticmethod
    def plot_train_val_accuracy(train_accuracies, val_accuracies, num_epochs):
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_val_loss(train_loss, val_loss, num_epochs):
       
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
        plt.legend()
        plt.show()


    @staticmethod
    def evaluate(all_targets, mlp_output, show_plot=False):
        
        mlp_output = mlp_output.cpu()
        all_targets = all_targets.cpu()
        predicted_labels = np.where(mlp_output.cpu() > 0.5, 1, 0)
        predicted_labels = predicted_labels.reshape(-1)
        all_predictions = predicted_labels

        fpr, tpr, thresholds = metrics.roc_curve(all_targets, mlp_output)
        auc = np.round(metrics.auc(fpr, tpr), 4)

        precision, recall, thresholds = metrics.precision_recall_curve(all_targets, mlp_output)
        auprc = np.round(metrics.auc(recall, precision), 4)

        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        f1_score = 2 * precision * recall / (precision + recall)

        print(f'Confusion matrix:\n{cm}')
        print(
            f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1_score:.3f}, AUC: {auc:.3f}, ,AUPRC: {auprc:.3f}')

        # Step 5: Display ROC and PR curves if requested
        if show_plot:
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: AUC={auc}')
            plt.plot(fpr, tpr)
            plt.show()

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR Curve: AUPRC={auprc}')
            plt.plot(recall, precision)
            plt.show()

            prediction_targets = pd.DataFrame({}, columns=['Prediction', 'Target'])

            res = pd.concat(
                [pd.DataFrame(mlp_output.numpy(), ), pd.DataFrame(all_targets.numpy())], axis=1,
                ignore_index=True)

            res.columns = prediction_targets.columns
            prediction_targets = pd.concat([prediction_targets, res])
            class_one = prediction_targets.loc[prediction_targets['Target'] == 0, 'Prediction']
            class_minus_one = prediction_targets.loc[prediction_targets['Target'] == 1, 'Prediction']

            fig, ax = plt.subplots()
            ax.set_ylabel("DeepDRA score")
            xticklabels = ['Responder', 'Non Responder']
            ax.set_xticks([1, 2])
            ax.set_xticklabels(xticklabels)
            data_to_plot = [class_minus_one, class_one]
            plt.ylim(0, 1)
            p_value = np.format_float_scientific(ttest_ind(class_one, class_minus_one)[1])
            cancer = 'all'
            plt.title(
                f'Responder/Non-responder scores for {cancer} cancer with \np-value ~= {p_value[0]}e{p_value[-3:]} ')
            bp = ax.violinplot(data_to_plot, showextrema=True, showmeans=True, showmedians=True)
            bp['cmeans'].set_color('r')
            bp['cmedians'].set_color('g')
            plt.show()

        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 score': f1_score, 'AUC': auc,
                'AUPRC': auprc}

    @staticmethod
    def add_results(result_list, current_result):
        result_list['AUC'].append(current_result['AUC'])
        result_list['AUPRC'].append(current_result['AUPRC'])
        result_list['Accuracy'].append(current_result['Accuracy'])
        result_list['Precision'].append(current_result['Precision'])
        result_list['Recall'].append(current_result['Recall'])
        result_list['F1 score'].append(current_result['F1 score'])
        return result_list

    @staticmethod
    def show_final_results(result_list):
        print("Final Results:")
        for i in range(len(result_list["AUC"])):
            accuracy = result_list['Accuracy'][i]
            precision = result_list['Precision'][i]
            recall = result_list['Recall'][i]
            f1_score = result_list['F1 score'][i]
            auc = result_list['AUC'][i]
            auprc = result_list['AUPRC'][i]

            print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1_score:.3f}, AUC: {auc:.3f}, ,AUPRC: {auprc:.3f}')

        avg_auc = np.mean(result_list['AUC'])
        avg_auprc = np.mean(result_list['AUPRC'])
        std_auprc = np.std(result_list['AUPRC'])
        avg_accuracy = np.mean(result_list['Accuracy'])
        avg_precision = np.mean(result_list['Precision'])
        avg_recal = np.mean(result_list['Recall'])
        avg_f1score = np.mean(result_list['F1 score'])
        print(
            f'AVG: Accuracy: {avg_accuracy:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recal:.3f}, F1 score: {avg_f1score:.3f}, AUC: {avg_auc:.3f}, ,AUPRC: {avg_auprc:.3f}')

        print(" Average AUC: {:.3f} \t Average AUPRC: {:.3f} \t Std AUPRC: {:.3f}".format(avg_auc, avg_auprc, std_auprc))
        return {'Accuracy': avg_accuracy, 'Precision': avg_precision, 'Recall': avg_recal, 'F1 score': avg_f1score, 'AUC': avg_auc,
                'AUPRC': avg_auprc}