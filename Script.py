import sensor
import image
import lcd
import KPU as kpu

lcd.init()
sensor.reset()

sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
#sensor.set_vflip(1)
sensor.run(1)
classes = ["face"]
task = kpu.load(0x200000)
a = kpu.set_outputs(task, 0, 7,7,30)
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
a = kpu.init_yolo2(task, 0.3, 0.3, 5, anchor)
while(True):
    img = sensor.snapshot()
    code = kpu.run_yolo2(task, img)
    if code:
        for i in code:
            if(i.value()>0):
                print(i.value())
                a = img.draw_rectangle(i.rect())

    a = lcd.display(img)
a = kpu.deinit(task)
