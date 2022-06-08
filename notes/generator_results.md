# Generator Results

Note หน้านี้จะบันทึกเกี่ยวกับ Performance ของ Generator Arch ต่างๆ ซึ่งจะมีการเปรียบเทียบโมเดล
ไม่กี่ Arch แต่จะเน้นหนักไปที่ KD แต่ละแบบว่าแบบไหนให้ผลที่ดีที่สุด

### แบบ Whole framework training
แบบนี้น่าจะน่าเชื่อถือกว่า เพราะว่าการจะวัดว่า Generator ออกมาดีหรือไม่ มันขึ้นกับ Model อื่นๆด้วย

| exp_code      | Arch Name                | Model size (MB) | min FID* | Notes |
|:--------------|--------------------------|-----------------|----------|-------|
| `whole.exp10` | Generator-DCGANs         | 17.570137       | 259.28   |       |
| `whole.exp11` | Generator-DeptSep-DCGANs | 1.139289        |          |       |
| `whole.exp12` | 3Layers-Generator        | 0.763225        |          |       |


### แบบ Train เฉพาะ Generator

แบบนี้คือ Pre-trained classifier มาก่อนแล้ว แล้วส่ง latent ที่ fmri extractor สร้างขึ้นมา
เอาไปให้ Generator gen รูปออกมา

| exp_code    | Arch Name                | Model size (MB) | min FID* | Notes |
|:------------|--------------------------|-----------------|----------|-------|
| `gen.exp01` | Generator-DCGANs         | 17.570137       | 669.07   |       |
| `gen.exp02` | Generator-DeptSep-DCGANs | 1.139289        | 368.82   |       |
| `gen.exp03` | 3Layers-Generator        | 0.763225 MB     | 700.72   |       |

#### **หมายเหตุ FID*** 
- โดยปกติแล้ว FID จะใช้ feature จาก pre-trained InceptionNet V3 (ต้องเป็น weight 
เฉพาะสำหรับคำนวน FID ด้วย) มาคิด mu กับ sigma
- แต่ว่า feature ที่ออกมา มีขนาดค่อนข้างใหญ่ (2,048) และค่อนข้างกินทรัพยากรตอนคำนวน ทำให้ 
FID* ของเราจึงเป็น feature จาก Image extractor ที่ based on alexnet และ pre-trained 
จาก fMRI_HC dataset... แต่ถ้าให้กลับไปใช้ InceptionNet V3 ก็ทำได้น้ะ มี code ยุ
- ดังนั้น จากนี้ไปของตั้งชื่อว่า _**AlexFID**_
