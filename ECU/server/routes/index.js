const express = require("express");
const router = express.Router();
const alertRoutes = require("../domains/alert");

router.use("/alert", alertRoutes);

module.exports = router;