# To do list

On each day

## 2022-06-03
- [ ] จัดการที่อยู่ของ model ให้อยู่ที่เดียวกัน
- [ ] ลอง run AlexNet for FID แล้วดูว่ายังโอเคมั้ย
- [ ] ทำ gen_kd ที่รวม Discriminator เข้าไปด้วย
    - นั่นหมายความว่า เราต้อง export Discriminator ของ Teacher ออกมาด้วย
    - ต้อง export ทุกโมเดลใน epoch เดียวกัน อย่าใช้คนละ epoch เพราะ parameter ที่ส่งต่อกัน อาจจะไม่รองรับกัน
      ทำให้ผลลัพธ์เพี้ยน

## 2022-06-02

- [x] ทำ AlexNet สำหรับ FID
- เนื่องจากว่า Olivier เหมือนจะรี Gourami หรือไม่ก็ไฟดับ


## 2022-06-01

- ไล่งานค้างให้เสร็จ

## 2022-05-31

- ตรวจสอบ FID
    - [x] ที่บอกว่า return Activation ออกมา มันอยู่ layer ไหนกันแน่?
        - อันนี้อาจจะไม่ต้องพยายามทำขนาดนั้น เพราะถ้า run จริง อาจจะใช้ pre-trained InceptionNet ไปเลย
- เขียน Script
    - [ ] Generator ที่มีการรวม Discriminator ด้วยตาม paper compressing GANs
    - [ ] Whole framework อาจจะต้องใช้ FIDOrig ด้วย เพราะ FID* มันอาจจะมั่วนิ่ม ไม่ถูกสุขลักษณะ

## 2022-05-30

- เขียน Script
    - [ ] สำหรับ Whole framework แต่เพิ่ม FID และ save model ออกมาด้วย
- ลองไปหาดู
    - [x] GANs KD เพราะเรายังไม่แน่ใจว่าเวลา KD Generator เขาทำยังไงกัน
        -
        ได้เปเปอร์นี้มา: [Compressing GANs using Knowledge Distillation](https://arxiv.org/pdf/1902.00159.pdf?ref=https://githubhelp.com)

## 2022-05-29

- เขียน Script
    - [x] สำหรับ Train Classifier
        - [x] ลองรัน SimpleFC 10 layers
        - [x] Export SimpleFC 4 layers สำหรับ Generator-KD
    - [x] สำหรับ Train Generator ปกติ (ไม่ใช่ Generator1Block)
    - [x] ทำ Generator-KD
    - [x] FID Metrics
        - ตอนนี้เข้าใจล้ะว่าจะ implement ได้ยังไง
- Not so important
    - [x] ปรับให้ Reporter
        - ให้เก็บ Stack ของ extra message
        - เพิ่ม Time stamp ของแต่ละ extra message ด้วย