from roboflow import Roboflow
from ultralytics import YOLO

def main():

    rf = Roboflow(api_key="CGCw1ZaTVuJecbNRz5zy")
    project = rf.workspace("ojass-workspace-dta4z").project("plutox-detector")
    version = project.version(2)
    dataset = version.download("yolov8")
                
                    
    model = YOLO("yolo26s.pt")

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=200,
        patience=50,        
        imgsz=640,
        device=0,
        # workers=0           
    )

if __name__ == "__main__":
    main()

