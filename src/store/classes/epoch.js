export default class Epoch {
    constructor(epoch_id, nth_epoch, train_loss, validation_loss, weight, iou_value, map_value, validation_result) {
        this.epoch_id = epoch_id;
        this.nth_epoch = nth_epoch;
        this.train_loss = train_loss;
        this.validation_loss = validation_loss;
        this.weight = weight;
        this.iou_value = iou_value;
        this.map_value = map_value;
        this.validation_result = validation_result;
    }
}
