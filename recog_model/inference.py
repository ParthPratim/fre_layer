from talos.utils.best_model import best_model, activate_model
import pickle

class ImageInferenceEngine:

    @staticmethod
    def do_infer(classifier_model=None,rgb=None):
        if rgb.shape == (150,150,3):
            if classifier_model not None:
                scan_object = None
                with open("model/"+classifier_model+".pkl","rb") as mf:
                    scan_object = pickle.load(mf)

                prediction = activate_model(scan_object, best_model(scan_object, 'val_acc', False)).predict(rgb)
                return prediction
